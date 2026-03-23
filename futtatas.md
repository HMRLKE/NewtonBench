# Futtatási Parancsok

## 1. Gyors lánc-ellenőrzés

Egyetlen, korlátozott konfiguráció fut le, majd automatikusan elkészül a log és a riport.

```bash
python run_pipeline.py --preset quick --model_name gpt41mini
```

Példa szűkített, explicit gyors tesztre:

```bash
python run_pipeline.py --preset quick --model_name gpt41mini --module m0_gravity --equation_difficulty easy --model_system vanilla_equation --law_version v0 --agent_backend vanilla_agent --trials 1 --noise 0.0
```

## 2. Teljes benchmark egy modellel

Az összes modul, difficulty, system és mindkét backend lefut; a `v_unchanged` kontrollverzió alapból ki van zárva, hogy a fő benchmark a 324-es tasktérhez igazodjon.

```bash
python run_pipeline.py --preset benchmark --model_name gpt41mini
```

## 3. Teljes benchmark a `configs/models.txt` modelleivel

```bash
python run_pipeline.py --preset benchmark
```

## 4. Kontrollverziók (`v_unchanged`) bevonása

```bash
python run_pipeline.py --preset benchmark --model_name gpt41mini --include_unchanged
```

## 5. Riport újragenerálása egy korábbi run alapján

Ha ismert a `run_tag`, a riport külön is újraépíthető:

```bash
python result_analysis/summarize_results.py --result_dir evaluation_results --output_dir outputs/pipeline_runs/MY_RUN_TAG/report --run_tag MY_RUN_TAG
```

## Kimenetek

Minden `run_pipeline.py` futás egy saját könyvtárat kap itt:

```text
outputs/pipeline_runs/<run_tag>/
```

Ide kerülnek:

- `RESULTS_INDEX.md`: rövid térkép a legfontosabb eredményfájlokhoz
- `pipeline.log`: a teljes konzol-log fájlba mentve
- `manifest.json`: a futás metaadatai és a végrehajtott parancsok
- `report/law_accuracy_summary.csv`: a legfontosabb law-szintű pontossági tábla
- `report/law_accuracy_summary.md`: a fenti markdown nézetben
- `report/results_by_trial.csv`: trial-szintű részletes eredmények
- `report/config_summary.csv`: logikai konfigurációk szerinti aggregátum
- `report/aggregated_trial_summary.csv`: leaderboard / eredménytábla
- `report/summary_report.md`: rövid markdown összefoglaló

A legutóbbi pipeline futás run tagje itt olvasható:

```text
outputs/pipeline_runs/LATEST_RUN.txt
```
