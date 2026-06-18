import unittest

import pandas as pd

from live.paper_guard import (
    DailyPaperGuardError,
    raise_on_guard_errors,
    validate_daily_inputs,
    validate_daily_result,
)


class PaperGuardTest(unittest.TestCase):
    def test_missing_target_price_is_error(self):
        issues = validate_daily_inputs(
            target_weights=pd.Series({"AAA": 0.5}),
            latest_prices=pd.Series({"BBB": 10.0}),
            run_date="2024-01-31",
            target_date="2024-01-31",
            price_date="2024-01-31",
        )

        self.assertIn("missing_price_for_target", {x.code for x in issues})
        with self.assertRaises(DailyPaperGuardError):
            raise_on_guard_errors(issues)

    def test_all_orders_blocked_is_warning(self):
        result = {
            "cash": 1000.0,
            "account_snapshot": {
                "cash": 1000.0,
                "market_value": 0.0,
                "total_asset": 1000.0,
                "n_positions": 0,
            },
            "positions": pd.DataFrame(columns=["symbol", "shares", "available_shares"]),
            "order_checks": pd.DataFrame(
                {
                    "symbol": ["AAA", "BBB"],
                    "check_status": ["BLOCK", "BLOCK"],
                }
            ),
            "paper_trades": pd.DataFrame(),
        }

        issues = validate_daily_result(result)

        self.assertEqual(["all_orders_blocked"], [x.code for x in issues])
        self.assertEqual("WARNING", issues[0].severity)


if __name__ == "__main__":
    unittest.main()
