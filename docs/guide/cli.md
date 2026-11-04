# Command line

The package installs the `mchermes` command.

## `mchermes run`

```bash
mchermes run decision.yaml                   # table + recommendation
mchermes run decision.yaml --json            # machine-readable output
mchermes run decision.yaml --plot chart.png --theme dark
mchermes run decision.yaml --html report.html --csv samples.csv
mchermes run decision.yaml --given "rent > 3300"
mchermes run decision.yaml -n 1000000 --seed 7 --criterion risk_averse
```

| Option | Meaning |
|---|---|
| `--json` | Print the full result as JSON (metrics, sensitivity, EVPI, EVPPI) |
| `--plot, -p PATH` | Save the distributions chart |
| `--html PATH` | Write the standalone HTML report |
| `--csv PATH` | Write every sample as CSV |
| `--given, -g EXPR` | Keep only futures where the condition holds |
| `--samples, -n N` | Override the number of samples |
| `--seed, -s N` | Override the seed |
| `--criterion, -c` | `expected_value`, `risk_averse` or `min_regret` |
| `--theme` | `light` or `dark` for charts |

## `mchermes sweep`

```bash
mchermes sweep decision.yaml VARIABLE START STOP [--steps 11] [--plot sweep.png]
```

Fixes `VARIABLE` at evenly spaced values from `START` to `STOP`, prints the expected outcome of each option and the best option at each value, and reports breakeven points.

## `mchermes validate`

Checks a file and runs a tiny simulation. It exits with code 1 and a located error message when something is wrong. This is useful in CI for repositories of decision files.

## `mchermes init`

Writes an example `decision.yaml` to start from (`--force` overwrites).

Every command exits with code 0 on success and 1 on a user error, with the message printed to stderr.
