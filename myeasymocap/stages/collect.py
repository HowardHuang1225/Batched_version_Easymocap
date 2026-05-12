import numpy as np
from tqdm import tqdm


# ── original CheckFramePerson preserved unchanged ────────────────────────────

class CheckFramePerson:
    def __init__(self, key) -> None:
        self.key = key
        self.pids = []
        self.frames = 0

    def __call__(self, keypoints3d, pids):
        k3d_, pid_ = [], []
        for i, pid in enumerate(pids):
            if pid not in self.pids:
                if self.frames == 0:
                    print('[{}]/{:06d} Add person {}'.format(self.__class__.__name__, self.frames, pid))
                    self.pids.append(pid)
                else:
                    continue
            k3d_.append(keypoints3d[i])
            pid_.append(pid)
        self.frames += 1
        k3d_ = np.stack(k3d_)
        return {
            'keypoints3d': k3d_,
            'pids': pid_,
        }


# ── original (kept for reference / fallback) ─────────────────────────────────

class CollectMultiPersonMultiFrame:
    def __init__(self, key, min_frame=10) -> None:
        self.key = key
        self.min_frame = min_frame

    def __call__(self, keypoints3d, pids):
        records = {}
        for frame in tqdm(range(len(pids)), desc='Reading'):
            pid_frame = pids[frame]
            for i, pid in enumerate(pid_frame):
                if pid not in records:
                    records[pid] = {'frames': [], 'keypoints3d': []}
                records[pid]['frames'].append(frame)
                records[pid]['keypoints3d'].append(keypoints3d[frame][i])
        remove_id = []
        for pid, record in records.items():
            print('[{}] Collect person {} with {} frames'.format(self.__class__.__name__, pid, len(record['frames'])))
            record['keypoints3d'] = np.stack(record['keypoints3d']).astype(np.float32)
            if len(record['frames']) < self.min_frame:
                remove_id.append(pid)
        for pid in remove_id:
            records.pop(pid)
        return {'results': records}


# ── new module ───────────────────────────────────────────────────────────────

class CollectMultiPersonMultiFrameWithGapFill:
    """Collect keypoints3d across frames with gap filling for missing main persons.

    Pipeline:
      (1) Pass-1: collect every (pid → frames, keypoints3d) record, no filtering.
      (2) Detect the number of main persons N as the mode of per-frame person counts.
      (3) Pick the N pids with the most frames as "main persons".
      (4) For each frame missing a main person, search short-lived pids that exist
          on that frame and assign each missing slot the spatially nearest one.
          Original pids are preserved at this stage.
      (5) Reassign pids to a contiguous 0..N-1 range using torso-centroid temporal
          consistency, so downstream sees stable person ids regardless of how the
          original tracker fragmented them.

    Args:
      key:         name of the keypoints field (kept for compatibility)
      min_frame:   min frame count to qualify as a main person directly. Pids with
                   fewer frames are treated as fragments and only used for filling.
      max_dist:    spatial distance threshold (meters) for accepting a fragment as
                   a fill for a missing main person. Beyond this it is ignored.
      conf_thresh: confidence threshold for a joint to count when computing torso
                   centroid.
      n_main:      override auto-detected N. Use when the scene's person count is
                   known (e.g. always 2 subjects).
    """

    # OpenPose body25 indices used as a stable spatial anchor for matching.
    # Two shoulders (2,5), two hips (9,12), pelvis (8). Head/limbs are excluded
    # because they are the most occlusion-prone and noisy under triangulation.
    # Class attribute (not module-level) to avoid namespace lookup issues
    # under EasyMocap's dynamic stage loader.
    TORSO_IDX = [2, 5, 8, 9, 12]

    def __init__(self, key, min_frame=10, max_dist=0.5,
                 conf_thresh=0.1, n_main=None,
                 merge_dist=0.3, max_overlap=20,
                 lookback=30) -> None:
        self.key = key
        self.min_frame = min_frame
        self.max_dist = max_dist
        self.conf_thresh = conf_thresh
        self.n_main = n_main
        # parameters for merging tracker identity switches:
        # two pids are merged if they overlap in some frames AND the mean spatial
        # distance during overlap is < merge_dist AND overlap is short (< max_overlap).
        self.merge_dist = merge_dist
        self.max_overlap = max_overlap
        # how many frames to walk backward / forward when locating a main pid's
        # last-known centroid for gap filling. If a main person is missing from
        # the start of the video, raise this until their first appearance is
        # within range (e.g. lookback=100 if first detection is around frame 80).
        self.lookback = lookback

    # -- helpers (class-local to avoid module namespace issues) -------------
    def _torso_centroid(self, kp3d, conf_thresh=0.1):
        """Return mean 3D position of confident torso joints, or None if too few."""
        torso = kp3d[self.TORSO_IDX]
        valid = torso[:, 3] > conf_thresh
        if valid.sum() < 2:
            return None
        return torso[valid, :3].mean(axis=0)

    @staticmethod
    def _mode_count(per_frame_counts):
        """Most-common nonzero count across frames. Avoids Counter to
        sidestep namespace issues in EasyMocap's dynamic stage loader."""
        nonzero = [c for c in per_frame_counts if c > 0]
        if not nonzero:
            return 0
        tally = {}
        for c in nonzero:
            tally[c] = tally.get(c, 0) + 1
        return max(tally.items(), key=lambda kv: kv[1])[0]

    @staticmethod
    def _count_distribution(values):
        """Return dict-like distribution of values, no Counter dependency."""
        d = {}
        for v in values:
            d[v] = d.get(v, 0) + 1
        return d

    # -- step 1 -------------------------------------------------------------
    def _collect_all(self, keypoints3d, pids):
        records = {}
        n_frames = len(pids)
        for frame in tqdm(range(n_frames), desc='Pass 1: collect all'):
            for i, pid in enumerate(pids[frame]):
                if pid not in records:
                    records[pid] = {'frames': [], 'keypoints3d': []}
                records[pid]['frames'].append(frame)
                records[pid]['keypoints3d'].append(keypoints3d[frame][i])
        for pid, rec in records.items():
            rec['keypoints3d'] = np.stack(rec['keypoints3d']).astype(np.float32)
        return records, n_frames

    # -- step 1.5 (new): merge tracker identity switches --------------------
    def _merge_split_pids(self, records):
        """Merge pid pairs that look like tracker identity switches.

        Heuristic: A and B describe the same person if they have overlapping
        frames, the mean spatial distance during overlap is small, and the
        overlap is short (a real second person would overlap continuously).

        On overlap frames we keep the keypoint with higher mean confidence.
        """
        pids = list(records.keys())
        # pre-compute centroids per pid per frame for cheap distance lookups
        centroids = {}
        for pid in pids:
            cmap = {}
            for k, f in enumerate(records[pid]['frames']):
                c = self._torso_centroid(records[pid]['keypoints3d'][k],
                                         self.conf_thresh)
                if c is not None:
                    cmap[f] = (c, k)
            centroids[pid] = cmap

        # union-find for transitive merges: if A↔B and B↔C, all three merge
        parent = {p: p for p in pids}
        def find(p):
            while parent[p] != p:
                parent[p] = parent[parent[p]]
                p = parent[p]
            return p
        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        # check every pair
        merged_pairs = []
        for i, a in enumerate(pids):
            for b in pids[i+1:]:
                fa = set(records[a]['frames'])
                fb = set(records[b]['frames'])
                overlap = fa & fb
                if not overlap or len(overlap) > self.max_overlap:
                    continue

                # mean distance during overlap
                dists = []
                for f in overlap:
                    if f in centroids[a] and f in centroids[b]:
                        ca, _ = centroids[a][f]
                        cb, _ = centroids[b][f]
                        dists.append(np.linalg.norm(ca - cb))
                if not dists:
                    continue
                mean_d = float(np.mean(dists))
                if mean_d < self.merge_dist:
                    union(a, b)
                    merged_pairs.append((a, b, len(overlap), mean_d))

        if not merged_pairs:
            print('[{}] No identity-switch merges'.format(self.__class__.__name__))
            return records

        # group pids by their root
        groups = {}
        for p in pids:
            r = find(p)
            groups.setdefault(r, []).append(p)

        for a, b, ov, d in merged_pairs:
            print('[{}] Merge candidate: pid {} ↔ pid {} '
                  '(overlap={} frames, mean_dist={:.3f}m)'.format(
                      self.__class__.__name__, a, b, ov, d))

        # build merged records: for each group with >1 member, fuse into one
        # Strategy: sort all members by frame count (descending), fill frames
        # in that order, first-come first-served. Each frame is covered by the
        # longest pid that has data for it.
        # Rationale: tracker association stability correlates with track length.
        # Longer pids have more consistent triangulation error structure across
        # frames, so prioritising them by length (not by per-frame confidence)
        # minimises pid-source switches and the resulting jitter.
        new_records = {}
        for root, members in groups.items():
            if len(members) == 1:
                new_records[root] = records[root]
                continue
            print('[{}] Merging group {} into pid {}'.format(
                self.__class__.__name__, members, root))

            # sort members by length descending; ties broken by pid for determinism
            ordered = sorted(members,
                             key=lambda m: (-len(records[m]['frames']), m))
            print('[{}]   priority order (longest first): {}'.format(
                self.__class__.__name__,
                [(m, len(records[m]['frames'])) for m in ordered]))

            # fill greedily: each member contributes only frames not yet claimed
            frame_to_kp = {}
            contributions = []   # for logging: (pid, n_contributed_frames)
            for m in ordered:
                added = 0
                for k, f in enumerate(records[m]['frames']):
                    if f not in frame_to_kp:
                        frame_to_kp[f] = records[m]['keypoints3d'][k]
                        added += 1
                contributions.append((m, added))

            sorted_frames = sorted(frame_to_kp.keys())
            kps = np.stack([frame_to_kp[f] for f in sorted_frames]).astype(np.float32)
            new_records[root] = {
                'frames': sorted_frames,
                'keypoints3d': kps,
            }
            print('[{}]   -> merged pid {}: {} frames total. Contribution per pid: {}'.format(
                self.__class__.__name__, root, len(sorted_frames),
                contributions))

        return new_records

    # -- step 2 -------------------------------------------------------------
    def _detect_n_main(self, pids, n_frames):
        if self.n_main is not None:
            print('[{}] Using user-specified n_main = {}'.format(
                self.__class__.__name__, self.n_main))
            return self.n_main
        counts = [len(pids[f]) for f in range(n_frames)]
        n = self._mode_count(counts)
        print('[{}] Detected n_main = {} (mode of per-frame counts; '
              'distribution: {})'.format(self.__class__.__name__, n,
                                         self._count_distribution(counts)))
        return n

    # -- step 3 -------------------------------------------------------------
    def _pick_main_pids(self, records, n_main):
        # sort pids by frame count descending; take top n_main with at least min_frame
        sorted_pids = sorted(records.keys(),
                             key=lambda p: -len(records[p]['frames']))
        main_pids = []
        for pid in sorted_pids:
            if len(main_pids) >= n_main:
                break
            if len(records[pid]['frames']) >= self.min_frame:
                main_pids.append(pid)
        if len(main_pids) < n_main:
            print('[{}] WARNING: only {} pids meet min_frame={}, requested {}.'
                  ' Filling will be limited.'.format(
                      self.__class__.__name__, len(main_pids),
                      self.min_frame, n_main))
        short_pids = [p for p in records if p not in set(main_pids)]
        for pid in main_pids:
            print('[{}] Main person pid={} with {} frames'.format(
                self.__class__.__name__, pid, len(records[pid]['frames'])))
        print('[{}] {} short pids available for gap filling'.format(
            self.__class__.__name__, len(short_pids)))
        return main_pids, short_pids

    # -- step 4 -------------------------------------------------------------
    def _build_per_frame_index(self, records):
        """frame_idx[f][pid] = index into records[pid]['keypoints3d'] for that frame."""
        idx = {}
        for pid, rec in records.items():
            for k, f in enumerate(rec['frames']):
                idx.setdefault(f, {})[pid] = k
        return idx

    def _last_known_centroid(self, pid, frame, records, frame_idx):
        """Walk backwards (then forwards) up to self.lookback frames to find
        the most recent valid torso centroid for pid. Returns None if nothing
        is found within range."""
        for delta in range(0, self.lookback + 1):
            for f in (frame - delta, frame + delta):
                if f < 0:
                    continue
                if f in frame_idx and pid in frame_idx[f]:
                    k = frame_idx[f][pid]
                    c = self._torso_centroid(records[pid]['keypoints3d'][k],
                                             self.conf_thresh)
                    if c is not None:
                        return c
        return None

    def _fill_gaps(self, records, main_pids, short_pids, n_frames):
        if not main_pids:
            return 0

        frame_idx = self._build_per_frame_index(records)
        n_filled = 0

        # Diagnostic: collect per-(missing_pid, reason) lists of unfilled frames
        # so we can print a compact summary at the end. Reasons:
        #   'no_short_at_all'      → no short pid exists on this frame
        #   'no_short_with_torso'  → short pid(s) exist but none has confident torso
        #   'no_target'            → main pid's last-known centroid not found in lookback
        #   'dist_too_far:X.XXm'   → nearest candidate is beyond max_dist
        diag_unfilled = {}   # mp -> list of (frame, reason, extra_dict)

        def _record(mp, frame, reason, extra=None):
            diag_unfilled.setdefault(mp, []).append((frame, reason, extra or {}))

        for f in tqdm(range(n_frames), desc='Pass 2: fill gaps'):
            present_main = [p for p in main_pids if p in frame_idx.get(f, {})]
            missing_main = [p for p in main_pids if p not in present_main]
            if not missing_main:
                continue

            # candidate short pids that exist on this frame
            candidates = [p for p in short_pids if p in frame_idx.get(f, {})] if short_pids else []
            if not candidates:
                for mp in missing_main:
                    _record(mp, f, 'no_short_at_all')
                continue

            # compute candidate centroids
            cand_centroids = {}
            for cp in candidates:
                k = frame_idx[f][cp]
                c = self._torso_centroid(records[cp]['keypoints3d'][k],
                                         self.conf_thresh)
                if c is not None:
                    cand_centroids[cp] = c
            if not cand_centroids:
                for mp in missing_main:
                    _record(mp, f, 'no_short_with_torso',
                            {'candidate_pids': candidates})
                continue

            # for each missing main person, find their last-known centroid and
            # match to nearest unclaimed candidate within max_dist
            for mp in missing_main:
                target = self._last_known_centroid(mp, f, records, frame_idx)
                if target is None:
                    _record(mp, f, 'no_target')
                    continue
                best_pid, best_dist, all_dists = None, self.max_dist, {}
                for cp, cc in cand_centroids.items():
                    d = float(np.linalg.norm(cc - target))
                    all_dists[cp] = d
                    if d < best_dist:
                        best_pid, best_dist = cp, d
                if best_pid is None:
                    nearest_pid = min(all_dists, key=all_dists.get) if all_dists else None
                    _record(mp, f, 'dist_too_far',
                            {'nearest_pid': nearest_pid,
                             'nearest_dist': all_dists.get(nearest_pid),
                             'max_dist': self.max_dist,
                             'all_dists': all_dists})
                    continue

                # transfer this frame's keypoints from short pid to main pid
                k = frame_idx[f][best_pid]
                kp = records[best_pid]['keypoints3d'][k]
                records[mp]['frames'].append(f)
                records[mp]['keypoints3d'] = np.concatenate(
                    [records[mp]['keypoints3d'], kp[None]], axis=0)
                # update frame index so this candidate is not reused
                frame_idx[f][mp] = len(records[mp]['frames']) - 1
                del cand_centroids[best_pid]
                n_filled += 1

        # main person frames are now unsorted (we appended late); sort each
        for mp in main_pids:
            order = np.argsort(records[mp]['frames'])
            records[mp]['frames'] = [records[mp]['frames'][i] for i in order]
            records[mp]['keypoints3d'] = records[mp]['keypoints3d'][order]

        print('[{}] Filled {} missing-main-person frames'.format(
            self.__class__.__name__, n_filled))

        # Diagnostic summary: which frames stayed unfilled and why
        if diag_unfilled:
            print('[{}] Unfilled frames per main pid:'.format(self.__class__.__name__))
            for mp, entries in diag_unfilled.items():
                # group by reason for compact reporting
                by_reason = {}
                for frame, reason, extra in entries:
                    by_reason.setdefault(reason, []).append((frame, extra))
                print('  pid {} : {} unfilled frame(s) total'.format(
                    mp, len(entries)))
                for reason, items in by_reason.items():
                    frames_only = [it[0] for it in items]
                    # compact frame range display
                    if len(frames_only) > 12:
                        sample = '{}, {}, ..., {} (total {})'.format(
                            frames_only[0], frames_only[1],
                            frames_only[-1], len(frames_only))
                    else:
                        sample = ', '.join(str(f) for f in frames_only)
                    print('    [{}] frames: {}'.format(reason, sample))
                    # for distance-rejected, show the actual numbers for first few
                    if reason == 'dist_too_far':
                        for frame, extra in items[:5]:
                            print('      frame {}: nearest pid {} at {:.3f}m '
                                  '(max_dist={:.3f}m)'.format(
                                      frame, extra['nearest_pid'],
                                      extra['nearest_dist'], extra['max_dist']))
        return n_filled

    # -- step 5 -------------------------------------------------------------
    def _reassign_pids(self, records, main_pids):
        """Renumber main pids to 0..N-1 in order of average X position
        (left-to-right in world coordinates). Fragments are dropped."""
        # compute mean X for each main pid for stable ordering
        avg_x = {}
        for mp in main_pids:
            kps = records[mp]['keypoints3d']
            xs = []
            for kp in kps:
                c = self._torso_centroid(kp, self.conf_thresh)
                if c is not None:
                    xs.append(c[0])
            avg_x[mp] = np.mean(xs) if xs else 0.0

        ordered = sorted(main_pids, key=lambda p: avg_x[p])
        final_records = {}
        for new_pid, old_pid in enumerate(ordered):
            rec = records[old_pid]
            print('[{}] Reassign: pid {} -> {} ({} frames, mean_x={:.3f})'.format(
                self.__class__.__name__, old_pid, new_pid,
                len(rec['frames']), avg_x[old_pid]))
            final_records[new_pid] = rec
        return final_records

    # -- entry --------------------------------------------------------------
    def __call__(self, keypoints3d, pids):
        records, n_frames = self._collect_all(keypoints3d, pids)

        # NEW step 1.5: merge tracker identity switches before counting persons
        records = self._merge_split_pids(records)

        # n_main from merged records (not raw pids list, which may overcount)
        per_frame_count = [0] * n_frames
        for pid, rec in records.items():
            for f in rec['frames']:
                per_frame_count[f] += 1
        if self.n_main is not None:
            n_main = self.n_main
            print('[{}] Using user-specified n_main = {}'.format(
                self.__class__.__name__, n_main))
        else:
            n_main = self._mode_count(per_frame_count)
            print('[{}] Detected n_main = {} after merging '
                  '(distribution: {})'.format(
                      self.__class__.__name__, n_main,
                      self._count_distribution(per_frame_count)))

        if n_main == 0:
            print('[{}] No persons detected'.format(self.__class__.__name__))
            return {'results': {}}
        main_pids, short_pids = self._pick_main_pids(records, n_main)
        self._fill_gaps(records, main_pids, short_pids, n_frames)
        final = self._reassign_pids(records, main_pids)

        # sanity report
        for pid, rec in final.items():
            print('[{}] Final person {} with {} frames (out of {})'.format(
                self.__class__.__name__, pid, len(rec['frames']), n_frames))

        return {'results': final}