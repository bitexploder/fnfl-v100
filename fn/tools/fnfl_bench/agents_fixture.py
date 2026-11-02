"""The agents bench's fixture: a tiny shop package with unittest tests, and six seeded easy tasks.

Each task seeds a clean copy (replacements in the base files, extra test files), gives the agent a prompt, and is graded
by its test files, rewritten from here before the run so the agent cannot change them. `fix` is a reference solution,
used only by selfcheck() to prove each task is solvable and fails before the fix."""

import os
import shutil
import subprocess
import sys
import tempfile

BASE = {
    "shop/__init__.py": "",
    "shop/pricing.py": '''def line_total(price, qty, discount=0.0):
    """Price of one order line: price * qty, less a fractional discount (0.1 = 10%)."""
    return round(price * qty * (1 - discount), 2)


def order_total(lines):
    return round(sum(line_total(l["price"], l["qty"], l.get("discount", 0.0)) for l in lines), 2)


def bulk_discount(qty):
    """10% off for 10 or more items."""
    return 0.1 if qty >= 10 else 0.0
''',
    "shop/inventory.py": '''def page(items, page_no, size):
    """The items on page page_no (pages start at 1)."""
    start = (page_no - 1) * size
    return items[start:start + size]


def low_stock(stock, threshold=5):
    """Sorted names of the items with fewer than threshold in stock."""
    return sorted(name for name, n in stock.items() if n < threshold)
''',
    "shop/config.py": '''DEFAULTS = {"currency": "EUR", "tax_rate": 0.2}


def get_setting(cfg, key):
    return cfg.get(key, DEFAULTS[key])
''',
    "shop/report.py": '''from shop import config, pricing


def format_lines(lines, cfg):
    cur = config.get_setting(cfg, "currency")
    out = []
    for line in lines:
        amount = pricing.line_total(line["price"], line["qty"], line.get("discount", 0.0))
        out.append(f"{line['name']}: {amount:.2f} {cur}")
    return out


def format_order(lines, cfg):
    cur = config.get_setting(cfg, "currency")
    total = pricing.order_total(lines)
    return f"Total: {total:.2f} {cur}"


def format_invoice(lines, cfg):
    cur = config.get_setting(cfg, "currency")
    total = pricing.order_total(lines)
    tax = round(total * config.get_setting(cfg, "tax_rate"), 2)
    return f"Net: {total:.2f} {cur}; tax: {tax:.2f} {cur}; gross: {total + tax:.2f} {cur}"
''',
    "shop/checkout.py": '''from shop import pricing


def checkout(lines, paid):
    """Change due, or raise ValueError when paid is too little."""
    due = pricing.order_total(lines)
    if paid < due:
        raise ValueError(f"need {due:.2f}, got {paid:.2f}")
    return round(paid - due, 2)
''',
    "tests/__init__.py": "",
    "tests/test_pricing.py": '''import unittest

from shop import pricing


class TestPricing(unittest.TestCase):
    def test_line_total(self):
        self.assertEqual(pricing.line_total(2.5, 4), 10.0)
        self.assertEqual(pricing.line_total(10, 1, 0.1), 9.0)

    def test_bulk_discount(self):
        self.assertEqual(pricing.bulk_discount(9), 0.0)
        self.assertEqual(pricing.bulk_discount(10), 0.1)
''',
    "tests/test_inventory.py": '''import unittest

from shop import inventory


class TestInventory(unittest.TestCase):
    def test_page(self):
        items = list(range(1, 11))
        self.assertEqual(inventory.page(items, 1, 3), [1, 2, 3])
        self.assertEqual(inventory.page(items, 4, 3), [10])

    def test_low_stock(self):
        stock = {"pear": 2, "apple": 9, "fig": 0, "kiwi": 5}
        self.assertEqual(inventory.low_stock(stock), ["fig", "pear"])
        self.assertEqual(inventory.low_stock(stock, 6), ["fig", "kiwi", "pear"])
''',
    "tests/test_report.py": '''import unittest

from shop import report

LINES = [{"name": "tea", "price": 4.0, "qty": 2}, {"name": "cup", "price": 10.0, "qty": 1, "discount": 0.5}]


class TestReport(unittest.TestCase):
    def test_format_lines(self):
        self.assertEqual(report.format_lines(LINES, {}), ["tea: 8.00 EUR", "cup: 5.00 EUR"])

    def test_format_order(self):
        self.assertEqual(report.format_order(LINES, {"currency": "USD"}), "Total: 13.00 USD")

    def test_format_invoice(self):
        self.assertEqual(report.format_invoice(LINES, {}), "Net: 13.00 EUR; tax: 2.60 EUR; gross: 15.60 EUR")
''',
    "tests/test_checkout.py": '''import unittest

from shop import checkout

LINES = [{"name": "tea", "price": 4.0, "qty": 2}]


class TestCheckout(unittest.TestCase):
    def test_change(self):
        self.assertEqual(checkout.checkout(LINES, 10), 2.0)

    def test_short(self):
        with self.assertRaises(ValueError):
            checkout.checkout(LINES, 5)
''',
    "README.md": "A tiny shop package. Run the tests with: python3 -m unittest discover -s tests -t .\n",
}

RENAME_TESTS = {"tests/test_rename.py": '''import unittest

from shop import checkout, pricing, report


class TestRename(unittest.TestCase):
    def test_cart_total(self):
        self.assertEqual(pricing.cart_total([{"price": 2.0, "qty": 3}]), 6.0)
        self.assertFalse(hasattr(pricing, "order_total"))

    def test_users(self):
        for mod in (report, checkout):
            src = open(mod.__file__).read()
            self.assertNotIn("order_total", src)
            self.assertIn("cart_total", src)
'''}

REFACTOR_TESTS = {"tests/test_refactor.py": '''import inspect
import unittest

from shop import report


class TestRefactor(unittest.TestCase):
    def test_helper(self):
        self.assertEqual(report.format_money(12.5, "EUR"), "12.50 EUR")

    def test_used(self):
        for fn in (report.format_order, report.format_invoice):
            src = inspect.getsource(fn)
            self.assertIn("format_money(", src)
            self.assertNotIn(":.2f}", src)
'''}

TASKS = [
    {"id": "off-by-one",
     "prompt": "The test tests/test_inventory.py::test_page fails: shop.inventory.page should return the items on a "
               "1-based page. Find and fix the bug. Run the tests with: python3 -m unittest discover -s tests -t .",
     "seed": {"shop/inventory.py": [("start = (page_no - 1) * size", "start = page_no * size")]},
     "fix": {"shop/inventory.py": [("start = page_no * size", "start = (page_no - 1) * size")]}},
    {"id": "misspelled-key",
     "prompt": "shop.report.format_order raises a KeyError and its test fails. Find and fix the bug. Run the tests "
               "with: python3 -m unittest discover -s tests -t .",
     "seed": {"shop/report.py": [('''    cur = config.get_setting(cfg, "currency")
    total = pricing.order_total(lines)
    return f"Total''', '''    cur = config.get_setting(cfg, "curency")
    total = pricing.order_total(lines)
    return f"Total''')]},
     "fix": {"shop/report.py": [('"curency"', '"currency"')]}},
    {"id": "missing-argument",
     "prompt": "shop.report.format_lines crashes with a TypeError and its test fails. Find and fix the bug. Run the "
               "tests with: python3 -m unittest discover -s tests -t .",
     "seed": {"shop/report.py": [('pricing.line_total(line["price"], line["qty"], line.get("discount", 0.0))',
                                  'pricing.line_total(line["price"], line.get("discount", 0.0))')]},
     "fix": {"shop/report.py": [('pricing.line_total(line["price"], line.get("discount", 0.0))',
                                 'pricing.line_total(line["price"], line["qty"], line.get("discount", 0.0))')]}},
    {"id": "add-function",
     "prompt": "Add the function shop.inventory.low_stock(stock, threshold=5): it takes a dict of item name to count "
               "and returns the sorted names whose count is below threshold. Its test in tests/test_inventory.py is "
               "already written. Run the tests with: python3 -m unittest discover -s tests -t .",
     "seed": {"shop/inventory.py": [('''

def low_stock(stock, threshold=5):
    """Sorted names of the items with fewer than threshold in stock."""
    return sorted(name for name, n in stock.items() if n < threshold)
''', "\n")]},
     "fix": {"shop/inventory.py": [("return items[start:start + size]\n",
                                    "return items[start:start + size]\n\n\ndef low_stock(stock, threshold=5):\n"
                                    "    return sorted(k for k, v in stock.items() if v < threshold)\n")]}},
    {"id": "rename",
     "prompt": "Rename the function shop.pricing.order_total to cart_total, and update every use of it (shop/pricing.py, "
               "shop/report.py and shop/checkout.py). Do not keep the old name. tests/test_rename.py checks it. Run the "
               "tests with: python3 -m unittest discover -s tests -t .",
     "seed": {}, "tests": RENAME_TESTS,
     "fix": {"shop/pricing.py": [("def order_total(", "def cart_total(")],
             "shop/report.py": [("pricing.order_total(", "pricing.cart_total(")] * 2,
             "shop/checkout.py": [("pricing.order_total(", "pricing.cart_total(")]}},
    {"id": "refactor",
     "prompt": "In shop/report.py, format_order and format_invoice repeat the money formatting (f\"{x:.2f} {cur}\"). "
               "Add a helper format_money(amount, currency) in shop/report.py that returns e.g. \"12.50 EUR\", and use it "
               "in both functions. Keep all tests green; tests/test_refactor.py checks the helper. Run the tests with: "
               "python3 -m unittest discover -s tests -t .",
     "seed": {}, "tests": REFACTOR_TESTS,
     "fix": {"shop/report.py": [("from shop import config, pricing\n",
                                 "from shop import config, pricing\n\n\ndef format_money(amount, currency):\n"
                                 "    return f\"{amount:.2f} {currency}\"\n"),
                                ('return f"Total: {total:.2f} {cur}"', 'return "Total: " + format_money(total, cur)'),
                                ('return f"Net: {total:.2f} {cur}; tax: {tax:.2f} {cur}; gross: {total + tax:.2f} {cur}"',
                                 'return f"Net: {format_money(total, cur)}; tax: {format_money(tax, cur)}; gross: '
                                 '{format_money(total + tax, cur)}"')]}},
]

TEST_CMD = [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-t", "."]


def _apply(root, edits):
    for path, reps in edits.items():
        p = os.path.join(root, path)
        s = open(p).read()
        for old, new in reps:
            if old not in s:
                raise RuntimeError(f"{path}: seed text not found: {old[:60]!r}")
            s = s.replace(old, new, 1)
        open(p, "w").write(s)


def _tests(task):
    return dict({k: v for k, v in BASE.items() if k.startswith("tests/")}, **task.get("tests", {}))


def make_copy(root, task):
    """A clean, seeded copy of the project for one task."""
    for path, text in BASE.items():
        os.makedirs(os.path.dirname(os.path.join(root, path)) or root, exist_ok=True)
        open(os.path.join(root, path), "w").write(text)
    _apply(root, task["seed"])
    for path, text in task.get("tests", {}).items():
        open(os.path.join(root, path), "w").write(text)


def grade(root, task, python=None, timeout=120, prefix=None):
    """Rewrite the task's tests from here, run them (under prefix, e.g. a sandbox): (passed, output)."""
    for path, text in _tests(task).items():
        os.makedirs(os.path.dirname(os.path.join(root, path)), exist_ok=True)
        open(os.path.join(root, path), "w").write(text)
    cmd = (prefix or []) + [python or sys.executable] + TEST_CMD[1:]
    try:
        p = subprocess.run(cmd, cwd=root, capture_output=True, text=True, timeout=timeout,
                           env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
        return p.returncode == 0, (p.stdout + p.stderr)[-3000:]
    except subprocess.TimeoutExpired:
        return False, "tests timed out"


def selfcheck():
    """Base passes; each seeded task fails; each reference fix passes."""
    ok = True
    with tempfile.TemporaryDirectory() as d:
        base = os.path.join(d, "base")
        make_copy(base, {"seed": {}})
        b, out = grade(base, {"seed": {}})
        print(f"base: {'pass' if b else 'FAIL'}")
        ok &= b
        for t in TASKS:
            r = os.path.join(d, t["id"])
            make_copy(r, t)
            before, _ = grade(r, t)
            _apply(r, t["fix"])
            after, out = grade(r, t)
            good = (not before) and after
            ok &= good
            print(f"{t['id']}: seeded {'pass' if before else 'fail'}, fixed {'pass' if after else 'fail'} -> "
                  f"{'ok' if good else 'BROKEN'}")
            if not after:
                print(out)
            shutil.rmtree(r)
    return ok


if __name__ == "__main__":
    sys.exit(0 if selfcheck() else 1)
