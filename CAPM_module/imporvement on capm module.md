You’re right to worry about scaling this up to 5 000+ calls—at that volume even seemingly “small” overheads will add up. Broadly speaking, I’d focus less on GPU or low‑level threading, and more on:

---

## 1. Eliminate redundant work

1. **Pre‑load & cache static data**  
   - You’re re‑reading and re‑parsing the market index and the risk‑free file on _every_ call. Instead, read, clean, and compute your index → percent‑changes and Rₓ daily series _once_ (at startup) and keep them in memory.  
   - Then in each `get_regression_line` call you only read the _single_ security file and merge against already‑prepared DataFrames.

2. **Avoid expensive Excel I/O in tight loop**  
   - `pd.read_excel` is pretty slow. If you can convert your master Excel archival to Parquet or even CSV up front, `pd.read_parquet` / `pd.read_csv` will be 5–10× faster.

---

## 2. Swap out statsmodels for a closed‑form beta

Fitting an OLS model via `statsmodels.OLS().fit()` has significant Python overhead (building model objects, parameter checks, summary tables, etc.).  For a CAPM regression you really only need:

\[
\beta = \frac{\sum (r_{m,t}-\bar r_m)(r_{i,t}-\bar r_i)}{\sum (r_{m,t}-\bar r_m)^2}
\quad,\quad
\alpha = \bar r_i - \beta \,\bar r_m
\]

A pure NumPy implementation looks like:

```python
import numpy as np

# given arrays excess_m, excess_i of shape (n,):
xm = excess_m - excess_m.mean()
xi = excess_i - excess_i.mean()
beta  = np.dot(xm, xi) / np.dot(xm, xm)
alpha = xi.mean() - beta * xm.mean()   # or: excess_i.mean() - beta*excess_m.mean()
```

This removes almost all Python‑level overhead and runs in pure C‑loops under the hood.

---

## 3. Parallelize at the “file” level

Since each security is completely independent, you can farm them out in parallel:

- **`multiprocessing.Pool`** (built in) or **`joblib.Parallel`**:  
  Divide your list of 5 000 filenames across 8–16 worker processes.  
  ```python
  from multiprocessing import Pool

  with Pool(processes=8) as pool:
      results = pool.map(get_regression_line, all_filenames)
  ```
- **Dask** or **Ray** if you want more resilience/finer control, but for pure embarrassingly‑parallel tasks `Pool` is usually enough.

Avoid threading here—because almost all the work (parsing, NumPy math) is CPU‑bound, the GIL will serialize your threads anyway.

---

## 4. When (and when not) to consider GPU

GPUs shine when you have very large matrix operations (e.g. millions × millions), but here your regressions are likely a few hundred to a few thousand daily observations. The overhead of copying each array to the device and back will swamp any speedup. Unless you’re sorting out tens of thousands of _time points_ per security, a GPU won’t help.

---

## 5. Putting it all together

1. **Startup**  
   ```python
   # – read and preprocess market & Rf once
   market_df = read_and_prep_index(...)
   rf_df     = read_and_prep_rf(...)
   merged_base = merge_market_rf(market_df, rf_df)
   ```
2. **Per‑security worker**  
   ```python
   def capm_for(sec_filename):
       sec_df = read_and_prep_security(sec_filename)
       df = pd.merge_asof(sec_df, merged_base, on='DATE', direction='backward')
       # compute percent changes & excess returns...
       β, α = closed_form_capm(df['excess_m'], df['excess_i'])
       return { 'id': ..., 'alpha': α, 'beta': β }
   ```
3. **Parallel dispatch**  
   ```python
   from multiprocessing import Pool
   with Pool() as p:
       all_results = p.map(capm_for, list_of_security_files)
   output_df = pd.DataFrame(all_results)
   ```

---

### Bottom line

- **Yes**, you do want to optimize if you’ll hit the 5 000‑call mark.
- **Don’t bother GPU**: your problem size (per‑regression) is too small.
- **Use vectorized closed‑form formulas** instead of `statsmodels` for speed.
- **Cache** everything that doesn’t change.
- **Parallelize** with `multiprocessing.Pool` (or `joblib`) rather than threads.

That combination should give you a 10×–100× speedup over your current loop of “read Excel → statsmodels” each time.