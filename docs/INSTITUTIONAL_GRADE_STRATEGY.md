# Institutional-Grade Meta Strategy

## Overview

A sophisticated meta-strategy designed to deliver **institutional-quality** risk-adjusted returns by combining multiple proven strategies with comprehensive risk management.

### Target Performance Profile

| Metric | Target | Category |
|--------|--------|----------|
| **Annualized Returns** | 12-18% | Elite Fund Performance |
| **Maximum Drawdown** | <12-15% | Capital Preservation |
| **Sharpe Ratio** | >1.5 | Risk-Adjusted Excellence |

This profile matches **top-50 elite multi-manager funds** and represents the sweet spot where institutional allocators seek consistent alpha without excessive volatility.

## Design Philosophy

### 1. Capital Preservation is King

**Key Insight**: Many allocators will fire managers after 20-25% drawdowns, regardless of past returns.

**Implementation**:
- Automatic de-risking at 12% drawdown (50% exposure reduction)
- Emergency protocols at 15% drawdown (25% exposure)
- Continuous drawdown monitoring and peak tracking
- Volatility targeting to maintain ~12% annualized vol

### 2. Consistency Over Home Runs

**Key Insight**: Funds with 25-40% returns but 30-60% drawdowns are viewed as too risky.

**Implementation**:
- Diversified strategy ensemble (4 complementary approaches)
- Monthly rebalancing to capture alpha while managing risk
- Correlation monitoring to prevent strategy convergence
- Regime-aware allocation adjustments

### 3. Multi-Strategy Diversification

**Ensemble Components** (Base Allocations):

```
Markowitz Portfolio:  40% — Core stability & mean-variance optimization
Momentum Rotation:    30% — Tactical alpha generation
Genetic-Regime:       20% — Adaptive risk with HMM protection
RSI Mean Reversion:   10% — Volatility dampening
```

**Why These Four?**

1. **Markowitz (40%)**: Proven Nobel-winning framework for portfolio construction. Provides stable core with optimal risk-return tradeoff.

2. **Momentum (30%)**: Exploits persistent trends in markets. Strong historical performance, especially in bull regimes.

3. **Genetic-Regime (20%)**: Combines genetic algorithm optimization with HMM-based regime detection. Automatically reduces exposure in bear markets (100% → 20%).

4. **RSI Mean Reversion (10%)**: Profits from short-term overreactions. Acts as portfolio stabilizer during high volatility.

### 4. Regime-Aware Risk Management

**Macro Context via HMM**:
- 2-state Hidden Markov Model trained on market returns
- Identifies bull vs bear regimes in real-time
- Adjusts portfolio exposure accordingly:
  - **Bull Regime**: 100% allocation (full exposure to alpha strategies)
  - **Bear Regime**: 70% allocation (defensive positioning)

## Risk Control Framework

### Level 1: Volatility Targeting

**Target**: 12% annualized volatility

**Mechanism**:
```julia
vol_scalar = target_vol / realized_vol  # Cap at 2x max
```

Dynamically scales positions to maintain consistent volatility profile.

### Level 2: Drawdown Protection

**Warning Level (12% DD)**:
- Reduce portfolio exposure to 50%
- Increase rebalancing frequency monitoring
- Alert: "Capital preservation mode activated"

**Emergency Level (15% DD)**:
- Reduce portfolio exposure to 25%
- Severely restrict new positions
- Alert: "Emergency risk reduction"

**Rationale**: Most institutional mandates have 15-20% drawdown limits before redemption/termination.

### Level 3: Correlation Management

**Threshold**: 0.7 correlation between strategies

**Action**: Blend toward equal-weighting (50% max adjustment)

**Why**: High correlation means diversification benefits are lost. When strategies converge, we shift toward equal allocation to restore diversification.

### Level 4: Regime-Based Scaling

**Bull Markets**: Full exposure (1.0x)

**Bear Markets**: Defensive (0.7x)

**Combined Risk Scalar**:
```julia
risk_scalar = drawdown_scalar × vol_scalar × regime_scalar
```

All factors multiply, meaning the most conservative constraint wins.

## Monthly Rebalancing

**Frequency**: Every 21 trading days (~monthly)

**Process**:
1. Evaluate recent performance of each strategy component
2. Calculate pairwise correlations between strategies
3. Adjust allocations if correlation >0.7 (move toward equal weight)
4. Recompute risk metrics and apply scaling
5. Execute rebalance

**Why Monthly?**
- Balances transaction costs vs. drift from target allocation
- Allows strategies sufficient time to deliver alpha
- Reduces noise from daily market fluctuations

## Expected Behavior Across Market Conditions

### Bull Market (2019-2021 style)
- **Exposure**: 100% (full allocation)
- **Primary Alpha**: Momentum (30%) and GeneticRegime (20%)
- **Expected**: 15-20% returns, 8-12% drawdowns
- **Sharpe**: 1.5-2.0

### Bear Market (2022 style)
- **Exposure**: 50-70% (defensive)
- **Primary Protection**: Drawdown controls + regime detection
- **Expected**: -5% to +3% returns, 10-15% drawdowns
- **Sharpe**: 0.0-0.5
- **Key**: Preserve capital while competition suffers 20-30% losses

### Sideways/Choppy Market (2018 style)
- **Exposure**: 70-100% (adaptive)
- **Primary Alpha**: RSI mean reversion (10%) + Markowitz stability
- **Expected**: 5-10% returns, 8-12% drawdowns
- **Sharpe**: 0.8-1.2

### Full Cycle (5-10 years)
- **Expected**: 12-18% annualized
- **Max DD**: 12-15%
- **Sharpe**: >1.5
- **Target**: Top-decile multi-manager fund performance

## Key Differentiators vs. Single Strategies

| Aspect | Single Strategy | Institutional-Grade |
|--------|----------------|-------------------|
| **Diversification** | Single approach | 4 complementary strategies |
| **Risk Management** | Basic stops | Multi-level: DD, vol, regime, correlation |
| **Adaptability** | Static allocation | Dynamic based on correlation & regime |
| **Drawdown Control** | Reactive | Proactive (warning → emergency levels) |
| **Market Cycles** | May fail in certain regimes | Designed for full-cycle performance |

## Usage

### Via CLI

```bash
# Compare against all strategies
julia --threads=4 ./bin/blab compare --stocks 10

# Test across multiple market periods (2018-2024)
julia --threads=4 ./bin/blab periods -n 10
```

The InstitutionalGrade strategy will be highlighted with special formatting showing its target metrics.

### Expected Output

```
======================================================================
Setting up INSTITUTIONAL-GRADE Meta Strategy...
Target: 12-18% returns, <15% drawdown, Sharpe >1.5
======================================================================

Training Institutional-Grade Meta Strategy...
======================================================================
Target Profile:
  • Returns: 12-18% annualized
  • Max Drawdown: <12-15%
  • Sharpe Ratio: >1.5
======================================================================

Using SPY for regime detection

======================================================================
TRAINING COMPONENT STRATEGIES
======================================================================

1/4 Training Markowitz Portfolio (Core Holdings)...
  [Markowitz training output...]

2/4 Training Momentum Rotation (Tactical Alpha)...
  [Momentum training output...]

3/4 Training Genetic-Regime (Adaptive Risk)...
  [GeneticRegime training output...]

4/4 Training RSI Mean Reversion (Volatility Dampening)...
  [RSI training output...]

======================================================================
TRAINING REGIME DETECTOR
======================================================================
HMM trained: μ=[0.08, -0.05]%, σ=[1.2, 2.3]%

======================================================================
BASE ALLOCATIONS:
======================================================================
  Markowitz: 40%
  Momentum: 30%
  GeneticRegime: 20%
  RSI: 10%

======================================================================
RISK CONTROLS:
======================================================================
  • Target Volatility: 12% annualized
  • Drawdown Warning (50% exposure): 12%
  • Drawdown Emergency (25% exposure): 15%
  • Correlation Threshold: 0.7
  • Rebalancing: Monthly (21 trading days)
======================================================================

✓ Institutional-Grade Meta Strategy training complete!
```

### During Backtesting

When risk controls activate, you'll see messages like:

```
Day 245: Risk scalar = 0.50 (DD: 12.3%, Vol: 15.2%, Regime: Bear)
Day 246: Risk scalar = 0.25 (DD: 15.1%, Vol: 18.7%, Regime: Bear)
```

This shows the strategy automatically de-risking to protect capital.

## Limitations & Considerations

### 1. Multi-Asset Requirement
- Requires at least 4-6 assets for Markowitz and Momentum
- Works best with 10+ diversified stocks
- Include SPY for optimal regime detection

### 2. Transaction Costs
- Monthly rebalancing generates moderate turnover
- Risk scaling can increase trading during volatile periods
- Consider in live trading: slippage, commissions, market impact

### 3. Market Regime Assumption
- HMM assumes market has distinct regimes (bull/bear)
- May struggle in transitional or unusual market conditions
- Regime detection has ~20-day lag (uses recent history)

### 4. Component Strategy Dependencies
- If underlying strategies fail to train, meta-strategy may fail
- Requires sufficient historical data for all components
- Training time is ~4x single strategy (runs 4 strategies in sequence)

### 5. Backtesting Idealization
- Assumes perfect execution at close prices
- No slippage, no partial fills, no market impact
- Real-world returns will be lower due to costs

## Performance Expectations

### Best Case (Strong Bull Market)
- Returns: 18-25%
- Drawdown: 8-10%
- Sharpe: 2.0+
- **Risk**: May underperform high-beta strategies in melt-ups

### Base Case (Mixed Market)
- Returns: 12-18%
- Drawdown: 12-15%
- Sharpe: 1.5-2.0
- **Target Profile**: Institutional quality

### Worst Case (Deep Bear Market)
- Returns: -5% to +5%
- Drawdown: 15-18%
- Sharpe: 0.0-0.5
- **Objective**: Preserve capital, beat -20% to -30% benchmarks

## Comparison to Hedge Fund Benchmarks

### HFRI Fund Weighted Composite Index
- **Returns**: ~7-10% annualized (10-year)
- **Drawdown**: ~15-20% (COVID crash)
- **Sharpe**: ~0.5-0.8

**InstitutionalGrade Target**: Meaningfully outperform on both returns AND risk-adjusted metrics.

### Top-Decile Multi-Strategy Funds
- **Returns**: 15-25% annualized
- **Drawdown**: <15%
- **Sharpe**: 1.7-2.0+

**InstitutionalGrade Target**: Match or exceed this elite cohort.

## Future Enhancements

### Potential Improvements

1. **Options Overlay**
   - Add protective puts during high-risk regimes
   - Sell covered calls in low-volatility periods
   - Target: Reduce tail risk, enhance Sharpe by 0.2-0.3

2. **Machine Learning Regime Detection**
   - Replace HMM with XGBoost or LSTM
   - Incorporate macro indicators (VIX, yield curve, etc.)
   - Target: Earlier regime change detection

3. **Dynamic Strategy Weights**
   - Use reinforcement learning for allocation
   - Optimize based on recent Sharpe, not just correlation
   - Target: Improve adaptive allocation by 10-15%

4. **Tail Risk Hedging**
   - Allocate 1-2% to VIX calls or put spreads
   - Activate only when implied vol is cheap
   - Target: Reduce max drawdown by 2-3%

5. **Multi-Asset Class Expansion**
   - Add bonds, commodities, crypto (small allocation)
   - True diversification beyond equities
   - Target: Reduce correlation to stock market by 0.1-0.2

## Conclusion

The **Institutional-Grade Meta Strategy** is designed from first principles to deliver the risk-return profile that elite allocators demand:

✅ **Consistent returns** (12-18%) without boom-bust cycles
✅ **Capital preservation** (<15% max drawdown)
✅ **Risk-adjusted excellence** (Sharpe >1.5)
✅ **Full-cycle performance** across bull, bear, and sideways markets

By combining proven strategies with sophisticated risk management, it targets the **top-decile of multi-manager hedge funds** — the rare territory where returns are strong, drawdowns are shallow, and capital flows freely.

---

**Remember**: In institutional asset management, **surviving the downturns** is more important than crushing the upturns. This strategy is built to compound capital safely over full market cycles, not to chase home-run years that blow up in bear markets.
