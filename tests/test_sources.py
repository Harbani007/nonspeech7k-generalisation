"""Unit tests for source identity, duplicate merging and leakage checks. Run: python -m unittest discover tests"""
import os
import sys
import unittest

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sources import KNOWN, INFERRED, effective_sources, leakage, merge_duplicates, namespaced_file_id, parse_source  # noqa: E402


class ParseSource(unittest.TestCase):
    def test_freesound_uses_sound_id(self):
        s = parse_source("108160-3_0_0.wav", "108160", "freesound")
        self.assertEqual((s.key, s.confidence), ("fs:108160", KNOWN))

    def test_youtube_recovers_truncated_id(self):
        self.assertEqual(parse_source("W7rJpRqK-8E(m)-100_3_0.wav", "W7rJpRqK", "youtube").key, "yt:W7rJpRqK-8E")
        self.assertEqual(parse_source("jOUp2ODyfns-12_0_0.wav", "jOUp2ODyfns", "youtube").key, "yt:jOUp2ODyfns")

    def test_aigei_youtube_name_shares_namespace_with_youtube_site(self):
        a = parse_source("142=KKKId0LDwVc-192_2_0.wav", "142=KKKId0LDwVc", "aigei")
        self.assertEqual((a.key, a.confidence), ("yt:KKKId0LDwVc", INFERRED))
        b = parse_source("5StJa6Ia6bI-3_2_0.wav", "5StJa6Ia6bI", "youtube")
        c = parse_source("9=5StJa6Ia6bI-19_2_0.wav", "9=5StJa6Ia6bI", "aigei")
        self.assertEqual(b.key, c.key)

    def test_youtube_id_containing_hyphen(self):
        self.assertEqual(parse_source("7=p73tJ6G-0-1-6_2_0.wav", "7", "aigei").key, "yt:p73tJ6G-0-1")

    def test_aigei_family_counters_do_not_collide(self):
        keys = {parse_source(f, "79", "aigei").key for f in
                ["79-cough-aigei-1_1_0.wav", "79-sneeze-aigei-1_5_0.wav"]}
        keys.add(parse_source("79_4_0.wav", "79", "freesound").key)
        self.assertEqual(len(keys), 3)

    def test_aigei_family_suffix_groups_together(self):
        a = parse_source("107-cough-aigei-1_1_0.wav", "107", "aigei").key
        b = parse_source("107-cough-aigei-5_1_0.wav", "107", "aigei").key
        c = parse_source("1-man yawn-aigei_6_0.wav", "1", "aigei").key
        d = parse_source("1-woman yawn-aigei_6_0.wav", "1", "aigei").key
        self.assertEqual(a, b)
        self.assertNotEqual(c, d)

    def test_aigei_numeric_key_is_candidate_not_merged_with_freesound(self):
        s = parse_source("172=196395-5_2_0.wav", "172=196395", "aigei")
        self.assertEqual((s.key, s.freesound_id_candidate), ("aigei:num:196395", "196395"))
        self.assertNotEqual(s.key, parse_source("196395-4_0_0.wav", "196395", "freesound").key)
        self.assertEqual(parse_source("148=17933_2_0.wav", "148=17933", "aigei").key, "aigei:num:17933")

    def test_aigei_crying_index(self):
        a = parse_source("50=13-man crying-aigei-9_2_0.wav", "50=13", "aigei").key
        b = parse_source("51=13-man crying-aigei_2_0.wav", "51=13", "aigei").key
        self.assertEqual(a, b)
        self.assertEqual(a, "aigei:man crying:13")


class Grouping(unittest.TestCase):
    def setUp(self):
        self.meta = pd.DataFrame({
            "filename": ["a1", "a2", "b1", "c1", "d1"],
            "group": ["A", "A", "B", "C", "D"],
            "split": ["train", "train", "test", "test", "train"],
        })

    def test_transitive_merge(self):
        pairs = pd.DataFrame({"a": ["a2", "b1"], "b": ["b1", "c1"]})
        g = merge_duplicates(self.meta["group"], self.meta["filename"], pairs)
        self.assertEqual(list(g), ["A", "A", "A", "A", "D"])

    def test_merge_is_order_independent(self):
        p1 = pd.DataFrame({"a": ["c1", "b1"], "b": ["b1", "a1"]})
        p2 = pd.DataFrame({"a": ["a1"], "b": ["c1"]})
        g1 = merge_duplicates(self.meta["group"], self.meta["filename"], p1)
        g2 = merge_duplicates(self.meta["group"], self.meta["filename"],
                              pd.concat([p2, pd.DataFrame({"a": ["b1"], "b": ["c1"]})]))
        self.assertEqual(list(g1), list(g2))

    def test_unknown_filename_raises(self):
        with self.assertRaises(KeyError):
            merge_duplicates(self.meta["group"], self.meta["filename"], pd.DataFrame({"a": ["zz"], "b": ["a1"]}))

    def test_leakage_detects_merged_cross_split_group(self):
        self.assertEqual(len(leakage(self.meta)), 0)
        m = self.meta.assign(group=merge_duplicates(self.meta["group"], self.meta["filename"],
                                                    pd.DataFrame({"a": ["a1"], "b": ["b1"]})))
        self.assertEqual(list(leakage(m).index), ["A"])

    def test_namespaced_file_id(self):
        m = pd.DataFrame({"filename": ["79-cough-aigei_1_0.wav", "79-sneeze-aigei_5_0.wav", "79_4_0.wav"],
                          "file_id": ["79", "79", "79"], "source": ["aigei", "aigei", "freesound"]})
        self.assertEqual(namespaced_file_id(m).nunique(), 3)

    def test_effective_sources(self):
        self.assertAlmostEqual(effective_sources([10, 10, 10]), 3.0)
        self.assertLess(effective_sources([376, 10, 10, 10]), 2.0)


if __name__ == "__main__":
    unittest.main()
