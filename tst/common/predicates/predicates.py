import unittest

from common.predicates.predicates import _evaluate_predicate as eval_p


class TestPredicates(unittest.TestCase):

    def test_logic(self):
        self.assertTrue(eval_p("0 * 0 < 1 + 1", {}, {}))
        self.assertTrue(eval_p("5/2.5 == 3 - 1", {}, {}))
        self.assertFalse(eval_p("0 * 0 >= 1 + 1", {}, {}))
        self.assertTrue(eval_p("1 + 2 * 3 < (1 + 2) * 3", {}, {}))
        self.assertTrue(eval_p("TRUE AND TRUE", {}, {}))
        self.assertTrue(eval_p("TRUE OR FALSE", {}, {}))
        self.assertTrue(eval_p("NOT FALSE AND (TRUE OR FALSE)", {}, {}))
        self.assertFalse(eval_p("NOT TRUE AND TRUE OR FALSE", {}, {}))
        self.assertTrue(eval_p("TRUE OR TRUE AND FALSE", {}, {}))
        self.assertFalse(eval_p("(TRUE OR TRUE) AND FALSE", {}, {}))

    def test_quantifiers(self):
        all_zero = [0, 0, 0, 0, 0]
        not_all_zero = [0, 0, 0, 1, 0]

        self.assertTrue(eval_p("ALL p: all_zero[p] == 0", {"all_zero": all_zero}, {}))
        self.assertFalse(eval_p("ALL p: not_all_zero[p] == 0", {"not_all_zero": not_all_zero}, {}))
        self.assertFalse(eval_p("EXISTS p: all_zero[p] == 1", {"all_zero": all_zero}, {}))
        self.assertTrue(eval_p("EXISTS p: not_all_zero[p] == 1", {"not_all_zero": not_all_zero}, {}))


    def test_functions(self):
        loc1 = [0, 0]
        loc2 = [2, 3]

        self.assertTrue(eval_p("manhattan_distance(loc1, loc2) == 5", {"loc1": loc1, "loc2": loc2}, {}))
        self.assertTrue(eval_p("manhattan_distance(loc1, loc2, 0) == -2", {"loc1": loc1, "loc2": loc2}, {}))
        self.assertTrue(eval_p("manhattan_distance(loc1, loc2, 3) == -3", {"loc1": loc1, "loc2": loc2}, {}))


if __name__ == '__main__':
    unittest.main()