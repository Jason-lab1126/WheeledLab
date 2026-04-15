# Analysis of Changes to Drift Environment Configs

## Summary of Changes

### `mushr_drift_env_cfg.py` Changes:

1. **New Reward Functions:**
   - `base_line_acc_x()` (lines 281-317): Computes longitudinal acceleration (x-component) by tracking velocity changes over time
   - `turn_left_go_right()` (lines 271-279): Rewards counter-steering behavior (steering opposite to angular velocity)

2. **New Observation Configuration (`HistoryCfg`):**
   - `history_length=20`: Adds temporal history to observations (20 timesteps)
   - `slip_angle_term`: Adds side slip angle observation to policy observations

3. **Existing Reward Structure:**
   - `side_slip`: Weight 10.0 (increases via curriculum)
   - `vel_dist`: Weight -5.0 (penalizes deviation from MAX_SPEED=3.0 m/s)
   - `progress`: Weight 40.0 (tracks angular velocity around track)
   - `tlgr`: Weight 0.0 initially (increases via curriculum)
   - `turn_energy`: Weight 20.0 (rewards speed² in corners)
   - `cross_track`: Weight -50.0 (penalizes distance from track center)
   - `term_pens`: Weight -5000.0 (termination penalty)

### `mushr_drift_recurrent_cfg.py` Changes:

1. **New Observation Configuration (`RecurrentObsCfg`):**
   - Designed for recurrent policies (history_length stays at 1)
   - Adds specific observations:
     - `wheel_ang_vel_term`: Wheel angular velocities
     - `base_angle_vel_term`: Base angular velocity
     - `base_lin_acc_x_term`: Longitudinal acceleration (uses `base_line_acc_x`)
     - `prev_action`: Previous action taken

2. **Inherits from base config:**
   - Uses same rewards (`DriftRewardsCfg`)
   - Uses same events (`DriftEventsRandomCfg`)
   - Uses same curriculum (`DriftCurriculumCfg`)

### Reset Behavior:
- `reset_root_state_along_track` (lines 132-133 in `mdp/events.py`): **Sets all velocities to ZERO** on reset

---

## Analysis: Are the Problems Addressed?

### ❌ Problem 1: "Deliberate Instability Injection" - **NOT ADDRESSED**
**Issue:** High initial kinetic energy → aggressive steering → rapid longitudinal deceleration

**Current State:**
- Line 133 in `mdp/events.py`: `asset.write_root_velocity_to_sim(torch.zeros((len(env_ids), 6), device=env.device), env_ids=env_ids)`
- **All velocities are set to zero on reset** - no initial kinetic energy injection
- No mechanism to start with high speed + steering input

**What's Needed:**
- Initialize with forward velocity near MAX_SPEED (e.g., 2.5-3.0 m/s)
- Add initial steering angle bias (e.g., ±0.3-0.5 rad)
- Optionally add initial angular velocity to induce instability

---

### ⚠️ Problem 2: "Policy Enters Drift Too Slowly" - **PARTIALLY ADDRESSED**
**Issue:** Policy doesn't enter drift quickly enough

**Current State:**
- `base_line_acc_x` observation added - policy can observe deceleration
- `turn_left_go_right` reward exists but starts at weight 0.0
- Curriculum increases `tlgr` weight over time (10.0 per 20 episodes, max 5 increases = 50.0)

**Gap:**
- No explicit reward for rapid drift entry
- No reward for high deceleration rate during entry phase
- `vel_dist` reward actually discourages maintaining high speed during aggressive maneuvers

**What's Needed:**
- Separate reward for rapid deceleration during drift entry
- Or modify `vel_dist` to only apply during sustained drift, not entry
- Higher initial weight for `tlgr` or separate "entry aggressiveness" reward

---

### ❌ Problem 3: "Policy Tries to Maintain Speed During Drift" - **NOT ADDRESSED**
**Issue:** Policy maintains speed during drift when it should allow deceleration

**Current State:**
- `vel_dist` reward (weight -5.0) penalizes any deviation from MAX_SPEED (3.0 m/s)
- This creates constant pressure to maintain speed
- `turn_energy` (weight 20.0) also rewards speed² in corners

**Problem:**
- Both rewards actively discourage speed loss, even during drift transitions
- No reward structure that explicitly allows/rewards controlled deceleration during drift

**What's Needed:**
- Modify `vel_dist` to have different targets based on drift phase
- Or add phase-aware reward that allows speed loss during entry/sustained drift
- Could use slip angle as a condition: if slip > threshold, relax speed requirement

---

### ⚠️ Problem 4: "Policy Avoids Sharp Movements" - **PARTIALLY ADDRESSED**
**Issue:** RL hates instability unless explicitly rewarded

**Current State:**
- `turn_left_go_right` reward exists but starts at 0.0 weight (curriculum increases it)
- `side_slip` reward (weight 10.0, increases to ~210.0 via curriculum) rewards slip angle
- No explicit reward for steering aggressiveness or rate of change

**Gap:**
- No reward for steering velocity/acceleration (sharp movements)
- Counter-steering (`tlgr`) reward is curriculum-gated, so early training avoids it
- No reward specifically for aggressive entry maneuvers

**What's Needed:**
- Reward for steering rate (how quickly steering changes)
- Higher initial weight for counter-steering behavior
- Separate "entry aggressiveness" reward for sharp initial steering

---

### ❌ Problem 5: "Current Reward Encourages Speed, Progress, Smooth Control" - **CONFIRMED ISSUE**
**Issue:** Reward structure biases toward smooth, fast, progress-oriented behavior

**Current State:**
- `progress` (40.0 weight): Strongly rewards angular velocity around track
- `vel_dist` (-5.0 weight): Penalizes speed deviation from MAX_SPEED
- `turn_energy` (20.0 weight): Rewards speed² in corners
- `cross_track` (-50.0 weight): Heavily penalizes track deviation (encourages smooth, predictable paths)

**Analysis:**
- These rewards create strong pressure for:
  - High, constant speed
  - Smooth track following
  - Predictable, non-aggressive maneuvers
- This directly conflicts with drift behavior which requires:
  - Variable speed (deceleration during entry)
  - Track overshoot (sideslip beyond track bounds)
  - Aggressive, unpredictable maneuvers

**What's Needed:**
- Reduce or condition `vel_dist` during drift phases
- Reduce `cross_track` penalty when slip angle is high (allows overshoot)
- Or restructure rewards to be phase-aware

---

### ❌ Problem 6: "Policy Doesn't Know Drift Phase" - **NOT ADDRESSED**
**Issue:** Policy doesn't know if it's entering, sustaining, or exiting drift

**Current State:**
- `mushr_drift_env_cfg.py` (`HistoryCfg`):
  - Has `slip_angle_term` observation
  - Has 20-step history, so policy can infer phase from temporal patterns
  - But no explicit phase indicators
  
- `mushr_drift_recurrent_cfg.py` (`RecurrentObsCfg`):
  - Has `base_lin_acc_x_term` (acceleration)
  - Has `wheel_ang_vel_term`, `base_angle_vel_term`
  - Has `prev_action`
  - History length = 1, so relies on recurrent network to track phase
  - But no explicit phase state

**Gap:**
- No explicit "drift phase" indicator (entering/sustaining/exiting)
- Phase must be inferred from:
  - Slip angle trajectory (in history)
  - Acceleration (from `base_lin_acc_x`)
  - Velocity changes
- This puts burden on network to learn phase detection

**What's Needed:**
- Explicit drift phase classification:
  - `drift_phase` observation: 0=straight, 1=entering, 2=sustaining, 3=exiting
  - Computed from slip angle, acceleration, and their derivatives
- Or add observations that make phase obvious:
  - `slip_angle_rate`: Rate of change of slip angle
  - `acceleration_magnitude`: Magnitude of deceleration
  - Binary flags: `is_entering_drift`, `is_in_drift`, `is_exiting_drift`

---

## Recommendations Summary

### Critical Fixes Needed:

1. **Add Initial Kinetic Energy Injection:**
   ```python
   # In reset_root_state_along_track, instead of zeros:
   initial_vel_x = torch.rand(len(env_ids), device=env.device) * (MAX_SPEED - 0.5) + 0.5  # 0.5-2.5 m/s
   initial_ang_vel_yaw = torch.randn(len(env_ids), device=env.device) * 0.3  # ±0.3 rad/s
   ```

2. **Add Phase-Aware Rewards:**
   - Condition `vel_dist` on drift phase (allow deceleration during entry)
   - Condition `cross_track` on slip angle (reduce penalty when slipping)

3. **Add Explicit Drift Phase Observations:**
   - Compute and expose drift phase state
   - Add slip_angle_rate, acceleration magnitude

4. **Increase Initial Aggressiveness Rewards:**
   - Start `tlgr` weight higher (e.g., 5.0-10.0 instead of 0.0)
   - Add steering rate reward for aggressive maneuvers

5. **Restructure Speed/Progress Rewards:**
   - Make `vel_dist` phase-aware or remove during drift phases
   - Reduce `progress` weight during entry phase (allows controlled deceleration)





