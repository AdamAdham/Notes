# DP-Notes

# Coin Change Problem

# DP-Notes

# Coin Change Problem

## Coin Change Problem Example with Walkthrough (Switching inner outer loops)

## Example:

- Suppose `n = 4` (we want to make 4 units of currency).
- Coins available: `coins = [1, 2]`.

---

## 1. **Original Implementation (Permutations):**

```go
for i = 1; i <= n; i++ {
    for coin := range coins {
        if i - coin >= 0 {
            dp[i] += dp[i - coin];
        }
    }
}
```

### Initialization:

- `dp = [1, 0, 0, 0, 0]` (since there's 1 way to make 0 units, which is to use no coins).

### Step-by-step Execution:

- **For `i = 1`:**

  - Loop over `coins = [1, 2]`.
    - For `coin = 1`:  
      `dp[1] += dp[1 - 1] => dp[1] += dp[0] => dp[1] = 1`
    - For `coin = 2`:  
      `i - coin < 0` so no update.
  - Result: `dp = [1, 1, 0, 0, 0]`

- **For `i = 2`:**

  - Loop over `coins = [1, 2]`.
    - For `coin = 1`:  
      `dp[2] += dp[2 - 1] => dp[2] += dp[1] => dp[2] = 1`
    - For `coin = 2`:  
      `dp[2] += dp[2 - 2] => dp[2] += dp[0] => dp[2] = 2`
  - Result: `dp = [1, 1, 2, 0, 0]`

- **For `i = 3`:**

  - Loop over `coins = [1, 2]`.
    - For `coin = 1`:  
      `dp[3] += dp[3 - 1] => dp[3] += dp[2] => dp[3] = 2`
    - For `coin = 2`:  
      `dp[3] += dp[3 - 2] => dp[3] += dp[1] => dp[3] = 3`
  - Result: `dp = [1, 1, 2, 3, 0]`

- **For `i = 4`:**
  - Loop over `coins = [1, 2]`.
    - For `coin = 1`:  
      `dp[4] += dp[4 - 1] => dp[4] += dp[3] => dp[4] = 3`
    - For `coin = 2`:  
      `dp[4] += dp[4 - 2] => dp[4] += dp[2] => dp[4] = 5`
  - Result: `dp = [1, 1, 2, 3, 5]`

### Final Result for Permutations:

- `dp[4] = 5` means there are **5 different permutations** of coins that sum to 4:
  - `{1, 1, 1, 1}`, `{1, 1, 2}`, `{1, 2, 1}`, `{2, 1, 1}`, `{2, 2}`.
  - You see that the different orders of `{1, 1, 2}` and `{2, 1, 1}` are counted separately, so this solution counts permutations.

---

## 2. **Switched Implementation (Combinations):**

```go
for coin := range coins {
    for i = 1; i <= n; i++ {
        if i - coin >= 0 {
            dp[i] += dp[i - coin];
        }
    }
}
```

### Initialization:

- `dp = [1, 0, 0, 0, 0]` (same starting point).

### Step-by-step Execution:

- **For `coin = 1`:**

  - Loop over `i = 1` to `4`.
    - For `i = 1`:  
      `dp[1] += dp[1 - 1] => dp[1] += dp[0] => dp[1] = 1`
    - For `i = 2`:  
      `dp[2] += dp[2 - 1] => dp[2] += dp[1] => dp[2] = 1`
    - For `i = 3`:  
      `dp[3] += dp[3 - 1] => dp[3] += dp[2] => dp[3] = 1`
    - For `i = 4`:  
      `dp[4] += dp[4 - 1] => dp[4] += dp[3] => dp[4] = 1`
  - Result: `dp = [1, 1, 1, 1, 1]`

- **For `coin = 2`:**
  - Loop over `i = 1` to `4`.
    - For `i = 1`:  
      `i - coin < 0`, so no update.
    - For `i = 2`:  
      `dp[2] += dp[2 - 2] => dp[2] += dp[0] => dp[2] = 2`
    - For `i = 3`:  
      `dp[3] += dp[3 - 2] => dp[3] += dp[1] => dp[3] = 2`
    - For `i = 4`:  
      `dp[4] += dp[4 - 2] => dp[4] += dp[2] => dp[4] = 3`
  - Result: `dp = [1, 1, 2, 2, 3]`

### Final Result for Combinations:

- `dp[4] = 3` means there are **3 distinct combinations** of coins that sum to 4:
  - `{1, 1, 1, 1}`, `{1, 1, 2}`, `{2, 2}`.
  - In this case, different orderings of `{1, 1, 2}` like `{2, 1, 1}` are **not counted separately**. Only unique combinations are counted.

---

## Conclusion:

- **Original implementation:** Counts permutations (order matters), so `dp[4] = 5`.
- **Switched implementation:** Counts combinations (order doesn’t matter), so `dp[4] = 3`.

### Personal Explanation

**1st example** For each number we get all combinations of coins, but **2nd example** we get in the first outer loop, the inner loop `for i in n` the total number of ways for `coins = [1]` so second inner loop we get the total number of ways for `coins = [1,2]` and so for example if `i=3` we only get `+=f(3-2) + f(3-2-1)` and not add `+=f(3-1) + f(3-1-2)` since at the point of decreasing the ones the coin 2 was not in the equation

**Another way of explanation:** `n=3` The first loop of coin=1 it will say `ways=[1,1,1,1]` for {1,1,1} and then coin=2 then get `ways[3] += ways=[3-2=1]` and `ways[2] += ways=[2-2=1]` so `ways=[1,1,2,2]`

This is why the loop order changes the result. When you iterate over the coins first, you're ensuring that each combination is only counted once, because you build up solutions using one coin at a time.
