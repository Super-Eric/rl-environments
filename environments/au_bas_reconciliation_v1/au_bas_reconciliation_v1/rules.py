"""The GST rules, applied to a ledger exactly as presented.

Kept separate from generation on purpose. If the answer key were whatever the generator
*intended* to inject, the environment would only ever test the generator's bookkeeping.
Instead the key is re-derived here by reading the finished ledger, so the model and the
grader are answering the same question from the same evidence — and a generator bug shows
up as a disagreement between the two (see `tests/test_au_bas.py`).

Rules implemented (Australian GST, as administered by the ATO):

- GST is 10%, so the GST inside a GST-inclusive price is `amount / 11`, rounded to the
  nearest cent (half up).
- `gst_free` supplies (basic food, most health and medical, education) carry no GST, and
  `input_taxed` supplies (financial supplies, residential rent) carry none and give no credit.
- An input tax credit on a purchase over $82.50 GST-inclusive requires a valid tax invoice
  showing the supplier's ABN; without one the credit is denied.
- Only documents dated inside the reporting quarter belong on that quarter's BAS.
- A re-issued invoice number is a duplicate; the later occurrence is not counted again.

The statutory source for each rule, and the simplifications this environment makes on top
of them, are listed under "Sources" in the package README.

BAS labels: G1 total sales (GST-inclusive), 1A GST on sales, 1B GST on purchases,
net = 1A - 1B (positive = payable to the ATO, negative = refund).
"""

from datetime import date
from decimal import ROUND_HALF_UP, Decimal

CENT = Decimal("0.01")
ITC_INVOICE_THRESHOLD = Decimal("82.50")  # GST-inclusive; $75.00 + 10% GST
GST_DIVISOR = Decimal("11")

GST_FREE_CATEGORIES = {"basic_food", "medical", "education", "export"}
INPUT_TAXED_CATEGORIES = {"financial_supply", "residential_rent"}

# Fault codes the model is asked to report. Names are part of the task contract.
FAULT_GST_ON_GST_FREE = "gst_on_gst_free"
FAULT_GST_MISCALCULATED = "gst_miscalculated"
FAULT_DUPLICATE_INVOICE = "duplicate_invoice"
FAULT_OUTSIDE_PERIOD = "outside_period"
FAULT_MISSING_ABN = "missing_abn_over_threshold"

ALL_FAULTS = (
    FAULT_GST_ON_GST_FREE,
    FAULT_GST_MISCALCULATED,
    FAULT_DUPLICATE_INVOICE,
    FAULT_OUTSIDE_PERIOD,
    FAULT_MISSING_ABN,
)


def money(value) -> Decimal:
    """Quantise to cents, half-up — the rounding the ATO specifies for GST amounts."""
    return Decimal(str(value)).quantize(CENT, rounding=ROUND_HALF_UP)


def correct_gst(amount: Decimal, category: str) -> Decimal:
    """The GST that *should* sit inside a GST-inclusive `amount` for `category`."""
    if category in GST_FREE_CATEGORIES or category in INPUT_TAXED_CATEGORIES:
        return money(0)
    return money(amount / GST_DIVISOR)


def audit(ledger: list[dict], period_start: str, period_end: str) -> dict:
    """Re-derive the corrected BAS and the fault list by reading `ledger`.

    Returns `{"faults": {txn_id: [codes]}, "bas": {"G1":.., "1A":.., "1B":.., "net":..}}`
    with every figure a string in cents-exact decimal form.
    """
    start, end = date.fromisoformat(period_start), date.fromisoformat(period_end)
    faults: dict[str, list[str]] = {}
    seen_invoices: dict[str, str] = {}
    g1 = gst_sales = gst_purchases = money(0)

    def flag(txn_id: str, code: str) -> None:
        faults.setdefault(txn_id, [])
        if code not in faults[txn_id]:
            faults[txn_id].append(code)

    for txn in ledger:
        tid, amount = txn["id"], money(txn["amount"])
        stated_gst, category = money(txn["gst"]), txn["category"]
        txn_date = date.fromisoformat(txn["date"])
        expected_gst = correct_gst(amount, category)

        # --- faults are independent of each other; a row can carry more than one ---
        out_of_period = not (start <= txn_date <= end)
        if out_of_period:
            flag(tid, FAULT_OUTSIDE_PERIOD)

        duplicate = txn["invoice_no"] in seen_invoices
        if duplicate:
            flag(tid, FAULT_DUPLICATE_INVOICE)
        else:
            seen_invoices[txn["invoice_no"]] = tid

        if stated_gst != expected_gst:
            # A GST-free or input-taxed line carrying GST is a different error from a
            # taxable line whose GST was simply computed wrong; graders and humans fix
            # them differently, so they get different codes.
            zero_rated = category in GST_FREE_CATEGORIES or category in INPUT_TAXED_CATEGORIES
            flag(tid, FAULT_GST_ON_GST_FREE if zero_rated else FAULT_GST_MISCALCULATED)

        missing_abn = txn["direction"] == "purchase" and amount > ITC_INVOICE_THRESHOLD and not txn.get("supplier_abn")
        if missing_abn:
            flag(tid, FAULT_MISSING_ABN)

        # --- corrected BAS: excluded rows contribute nothing, kept rows use correct GST ---
        if out_of_period or duplicate:
            continue
        if txn["direction"] == "sale":
            g1 += amount
            gst_sales += expected_gst
        elif not missing_abn and category not in INPUT_TAXED_CATEGORIES:
            gst_purchases += expected_gst

    return {
        "faults": faults,
        "bas": {
            "G1": str(money(g1)),
            "1A": str(money(gst_sales)),
            "1B": str(money(gst_purchases)),
            "net": str(money(gst_sales - gst_purchases)),
        },
    }
