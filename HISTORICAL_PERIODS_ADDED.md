# Historical Crisis Periods Added

**Date**: December 27, 2025
**Update**: Added 2000 Dot-com Bubble and 2008 Financial Crisis periods

---

## Data Coverage Verification

### Full 30-Stock Universe Coverage

**2000 Dot-com Bubble** (requires data before 2002-12-31):
- **Coverage**: 22/30 stocks (73%)
- **Available stocks**: AAPL, ADBE, AMZN, BA, BAC, CAT, CVX, HD, JNJ, JPM, KO, LLY, MCD, MRK, MSFT, NEE, NVDA, PG, UNH, UNP, WMT, XOM

**2008 Financial Crisis** (requires data before 2009-12-31):
- **Coverage**: 27/30 stocks (90%)
- **Available stocks**: All 22 from above PLUS AVGO, CRM, GOOGL, MA, V

**Recent only** (post-2009):
- **Count**: 3/30 stocks (10%)
- **Stocks**: ABBV (2013), META (2012), TSLA (2010)

---

## Periods Added to TimePeriodsAnalysis.jl

### 1. Dot-com Bubble Crash (2000-2002)

```julia
MarketPeriod(
    "Dotcom_Bubble_2000",
    DateTime("1997-01-01"),  # Train start
    DateTime("1999-01-01"),  # Test start (before peak March 2000)
    DateTime("2002-12-01"),  # Test end (after crash)
    "Dot-com Bubble Crash (2000-2002)"
)
```

**Historical context**:
- **Peak**: March 2000 (NASDAQ ~5,000)
- **Bottom**: October 2002 (NASDAQ ~1,100) - 78% decline
- **Duration**: ~2.5 years
- **Cause**: Internet/tech overvaluation, irrational exuberance

**Key characteristics**:
- Tech stocks crashed 80-90%
- Defensive sectors (healthcare, staples) held up relatively well
- Energy and financials also declined but less severely
- Recovery took years (NASDAQ didn't reclaim 2000 peak until 2015)

**Testing value**:
- Tests strategy performance in severe tech crash
- Validates sector diversification benefits
- Shows importance of defensive sectors
- Demonstrates value stocks vs growth stocks divergence

---

### 2. Financial Crisis (2007-2009)

```julia
MarketPeriod(
    "Financial_Crisis_2008",
    DateTime("2005-01-01"),  # Train start
    DateTime("2007-01-01"),  # Test start (before Bear Stearns)
    DateTime("2009-06-01"),  # Test end (after March 2009 bottom)
    "Financial Crisis (2007-2009)"
)
```

**Historical context**:
- **Peak**: October 2007 (S&P 500 ~1,565)
- **Bottom**: March 2009 (S&P 500 ~677) - 57% decline
- **Duration**: ~1.5 years
- **Cause**: Subprime mortgage crisis, credit freeze, banking collapse

**Key characteristics**:
- Financials crashed 80%+ (JPM, BAC, etc.)
- All sectors declined ("correlations → 1")
- Energy crashed with oil collapse
- Only staples and healthcare held up marginally better
- VIX hit 80+ (extreme volatility)

**Testing value**:
- Tests strategy in systemic crisis (all sectors down)
- Validates risk management and drawdown controls
- Shows whether defensive sectors provide any protection
- Tests regime detection during extreme volatility
- Demonstrates importance of correlation in crises

---

## New Testing Framework: 8 Periods (2000-2024)

### Complete Timeline

1. **Dotcom_Bubble_2000** (1997-2002) - 22 stocks
2. **Financial_Crisis_2008** (2005-2009) - 27 stocks
3. **2018_Correction** (2017-2019) - 30 stocks
4. **COVID_Crash** (2019-2020) - 30 stocks
5. **Bull_2020_2021** (2019-2021) - 30 stocks
6. **Rate_Hikes_2022** (2020-2022) - 30 stocks
7. **AI_Boom_2023** (2021-2023) - 30 stocks
8. **Recent_2024** (2022-2024) - 30 stocks

### Crisis Coverage

**Severe Crashes** (>40% decline):
- 2000 Dot-com: NASDAQ -78%
- 2008 Financial Crisis: S&P -57%

**Moderate Corrections** (15-30%):
- 2018 Q4 Correction: -19.8%
- 2020 COVID Crash: -34%
- 2022 Rate Hikes: Tech -30%+

**Bull Markets**:
- 2020-2021 Recovery: +100%+ from bottom
- 2023 AI Boom: +25%

---

## Expected Insights from Historical Periods

### 2000 Dot-com Bubble

**Hypothesis**: Cross-industry diversification will show massive benefits

**Expected results**:
- Tech-heavy portfolios (70% tech): -60% to -70% DD
- Diversified portfolios (27% tech + defensives): -30% to -40% DD
- Healthcare/Staples should provide strong protection
- Momentum strategies should fail (no uptrends)
- Mean reversion (RSI) may work better

**Key test for**:
- GeneticPortfolio (will likely overweight tech → massive DD)
- InstitutionalGrade (defensive allocation should help)
- AdaptiveRegime (regime detection + sector rotation)
- Markowitz (diversification benefits)

---

### 2008 Financial Crisis

**Hypothesis**: All strategies will struggle ("correlations → 1")

**Expected results**:
- Even diversified portfolios: -40% to -50% DD
- Financials destroyed: -80%+
- Energy crushed: -60%+
- Only healthcare/staples down -30% to -40%
- Cash positions become critical

**Key test for**:
- Volatility scaling (should reduce exposure as vol spikes)
- Drawdown controls (emergency 25% exposure)
- Regime detection (must identify bear market fast)
- Risk parity approaches (may fail as correlations spike)

**Institutional survival**:
- Strategies with >50% DD will "fail" institutional test
- Target: Keep DD under 40% even in 2008

---

## Theoretical Optimum (Updated)

### Previous Optimum (6 Periods, 2018-2024)

```
Returns:         15-25% annualized
Average DD:      8-15%
Worst DD:        15-20% (any period)
Sharpe:          1.1-1.3
```

**Best performers**:
- InstitutionalGrade: 20.8% returns, 16.6% worst DD, 1.14 Sharpe ✓
- AdaptiveRegime: 15.1% returns, 18.3% worst DD, 1.29 Sharpe ✓

---

### New Optimum (8 Periods, 2000-2024) - EXPECTED

```
Returns:         12-20% annualized (lower due to 2000/2008)
Average DD:      12-18%
Worst DD:        30-40% (2000 or 2008 will be worst)
Sharpe:          0.9-1.2
```

**Rationale**:
- 2000 and 2008 will dominate worst DD metric
- Even best strategies will see 30-40% DD in these crises
- BUT this is acceptable for institutional quality (< 50% threshold)
- Lower Sharpe is expected when including severe crashes

**Institutional acceptance**:
- <50% worst DD: Acceptable (survived worst crises)
- 30-40% worst DD: Good (better than market -57% to -78%)
- <30% worst DD: Excellent (rare for equity strategies)

---

## Success Criteria

### Passing Grade (Institutional Quality)

**Across all 8 periods**:
1. Worst DD < 50% (must survive 2008 without blowing up)
2. Average DD < 20% (controlled risk in normal periods)
3. Positive returns over full cycle (beat inflation)
4. Sharpe > 0.8 (reasonable risk-adjusted returns)

**Best in class**:
1. Worst DD < 40% (beat market in 2000/2008)
2. Average DD < 15%
3. Returns > 10% annualized
4. Sharpe > 1.0

---

## What We're Testing

1. **Does cross-industry diversification help in 2000?**
   - Tech crash should validate healthcare/staples/utilities
   - 27% tech vs 70% tech should show huge DD difference

2. **Do any strategies survive 2008 intact?**
   - S&P -57%, so <40% DD is "winning"
   - Risk controls (volatility scaling, DD triggers) must activate

3. **Which strategy has lowest worst-case DD?**
   - This is the ultimate institutional survival metric
   - Strategy that keeps job through 2000 AND 2008 wins

4. **Is InstitutionalGrade still best?**
   - Multi-level risk controls designed for crisis survival
   - Should outperform in 2008 (drawdown protection)
   - May lag in 2000 if not enough defensive allocation

5. **Does AdaptiveRegime's regime detection work in real crises?**
   - Must detect bear regime early in 2000/2008
   - Rotate to defensive strategies (RSI, staples)
   - Could be best performer if detection is fast

---

## Running the Analysis

```bash
# Test all strategies across 8 periods (2000-2024) with 30 stocks
julia --threads=4 ./bin/blab periods

# This will test:
# - 23 strategies
# - 8 market periods (including 2000, 2008)
# - 22-30 stocks (depending on period)
# - Full train/val/test splits
```

**Expected runtime**: 10-15 minutes (more periods + more training data)

---

## Next Steps

1. **Run full 8-period analysis** ✓ (in progress)
2. **Analyze results**:
   - Which strategy has lowest worst DD?
   - How bad are 2000/2008 drawdowns?
   - Does diversification help in 2000?
   - Do risk controls save strategies in 2008?

3. **Update documentation** with findings
4. **Adjust strategies if needed**:
   - If all strategies fail 2008 (>50% DD): Need more defensive allocation
   - If 2000 shows no diversification benefit: Investigate why
   - If momentum works in crashes: Update priors

5. **Final recommendation**:
   - Best strategy for 25-year backtest (2000-2024)
   - Deployment-ready institutional strategy

---

*Analysis framework updated December 27, 2025 - Blab Backtesting Library*
