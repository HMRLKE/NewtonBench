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

Provider-specifikus modellfájllal, a default `configs/models.txt` átírása nélkül:

```bash
python run_pipeline.py --preset benchmark --api_source g4s --models_file configs/models_g4s.txt
```

Ez a javasolt megoldás GenAI4Science többmodelles futásokhoz.

A repóban ehhez már benne van:

```text
configs/models_g4s.txt
```

Tartalma:

```text
gemma4:31b
deepseek-r1:32b
llama3.3:70b
gpt-oss:120b
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

## 6. Konzisztens vs. nem konzisztens összehasonlítás

Példa egy `v0`-ra szűrt benchmark-szeletre, ugyanazzal a modell-, task- és promptbeállítással.

Először az inkonzisztens futás:

```bash
python run_pipeline.py --preset benchmark --model_name gpt41mini --module m0_gravity --equation_difficulty easy --model_system vanilla_equation --law_version v0 --agent_backend vanilla_agent --prompt_set modified --run_tag v0-compare-inconsistent
```

Utána a konzisztens futás:

```bash
python run_pipeline.py --preset benchmark --model_name gpt41mini --module m0_gravity --equation_difficulty easy --model_system vanilla_equation --law_version v0 --agent_backend vanilla_agent --prompt_set modified --consistency --run_tag v0-compare-consistent
```

Végül az összehasonlító táblák elkészítése:

```bash
python result_analysis/compare_consistency.py --result_dir evaluation_results --output_dir outputs/pipeline_runs/v0-consistency-compare/report --inconsistent_run_tag v0-compare-inconsistent --consistent_run_tag v0-compare-consistent
```

Az összehasonlító kimenetek:

- `outputs/pipeline_runs/v0-consistency-compare/report/consistency_comparison.csv`
- `outputs/pipeline_runs/v0-consistency-compare/report/consistency_comparison.md`
- `outputs/pipeline_runs/v0-consistency-compare/report/consistency_model_summary.csv`
- `outputs/pipeline_runs/v0-consistency-compare/report/consistency_report.md`

## 7. Original vs. Modified prompt szett + consistency egyetlen táblában

Ha ugyanazt a benchmarkot mind a négy kombinációban akarod összehasonlítani:

- `original` + `inconsistent`
- `original` + `consistent`
- `modified` + `inconsistent`
- `modified` + `consistent`

akkor futtasd ugyanazzal a sweep-konfigurációval a négy runt, eltérő `run_tag`-ekkel.

Az `original` párosra példa:

```bash
python run_pipeline.py --preset benchmark --model_name gpt41mini --include_unchanged --prompt_set original --run_tag all-laws-inconsistent
python run_pipeline.py --preset benchmark --model_name gpt41mini --include_unchanged --prompt_set original --consistency --run_tag all-laws-consistent
```

Az ehhez illeszkedő `modified` páros:

```bash
python run_pipeline.py --preset benchmark --model_name gpt41mini --include_unchanged --prompt_set modified --run_tag all-laws-modified-inconsistent
python run_pipeline.py --preset benchmark --model_name gpt41mini --include_unchanged --prompt_set modified --consistency --run_tag all-laws-modified-consistent
```

Végül az egyetlen, négydimenziós összesítő tábla:

```bash
python result_analysis/compare_prompt_consistency.py --result_dir evaluation_results --output_dir outputs/pipeline_runs/all-laws-prompt-consistency-compare/report --original_inconsistent_run_tag all-laws-inconsistent --original_consistent_run_tag all-laws-consistent --modified_inconsistent_run_tag all-laws-modified-inconsistent --modified_consistent_run_tag all-laws-modified-consistent
```

A fő kimenetek:

- `outputs/pipeline_runs/all-laws-prompt-consistency-compare/report/prompt_consistency_comparison.csv`
- `outputs/pipeline_runs/all-laws-prompt-consistency-compare/report/prompt_consistency_model_summary.csv`
- `outputs/pipeline_runs/all-laws-prompt-consistency-compare/report/prompt_consistency_report.md`

## 8. Teljes OpenAI vs. GenAI4Science összehasonlító workflow

### 8.1. Egyetlen OpenAI modell, minden prompt- és consistency-kombináció

```bash
python run_pipeline.py --preset benchmark --model_name gpt41mini --api_source oa --include_unchanged --prompt_set original --run_tag oa-gpt41mini-original-inconsistent
python run_pipeline.py --preset benchmark --model_name gpt41mini --api_source oa --include_unchanged --prompt_set original --consistency --run_tag oa-gpt41mini-original-consistent
python run_pipeline.py --preset benchmark --model_name gpt41mini --api_source oa --include_unchanged --prompt_set modified --run_tag oa-gpt41mini-modified-inconsistent
python run_pipeline.py --preset benchmark --model_name gpt41mini --api_source oa --include_unchanged --prompt_set modified --consistency --run_tag oa-gpt41mini-modified-consistent
```

Összesítés:

```bash
python result_analysis/compare_prompt_consistency.py --result_dir evaluation_results --output_dir outputs/pipeline_runs/oa-gpt41mini-prompt-consistency/report --original_inconsistent_run_tag oa-gpt41mini-original-inconsistent --original_consistent_run_tag oa-gpt41mini-original-consistent --modified_inconsistent_run_tag oa-gpt41mini-modified-inconsistent --modified_consistent_run_tag oa-gpt41mini-modified-consistent
```

### 8.2. Több G4S modell külön modellfájlból

Hozz létre egy külön fájlt, például:

```text
configs/models_g4s.txt
```

amiben csak G4S-en elérhető modellek vannak, például:

```text
llama3.1:8b
qwen2.5:14b
deepseek-r1:32b
```

Ezután:

```bash
python run_pipeline.py --preset benchmark --api_source g4s --models_file configs/models_g4s.txt --include_unchanged --prompt_set original --run_tag g4s-original-inconsistent
python run_pipeline.py --preset benchmark --api_source g4s --models_file configs/models_g4s.txt --include_unchanged --prompt_set original --consistency --run_tag g4s-original-consistent
python run_pipeline.py --preset benchmark --api_source g4s --models_file configs/models_g4s.txt --include_unchanged --prompt_set modified --run_tag g4s-modified-inconsistent
python run_pipeline.py --preset benchmark --api_source g4s --models_file configs/models_g4s.txt --include_unchanged --prompt_set modified --consistency --run_tag g4s-modified-consistent
```

Összesítés:

```bash
python result_analysis/compare_prompt_consistency.py --result_dir evaluation_results --output_dir outputs/pipeline_runs/g4s-prompt-consistency/report --original_inconsistent_run_tag g4s-original-inconsistent --original_consistent_run_tag g4s-original-consistent --modified_inconsistent_run_tag g4s-modified-inconsistent --modified_consistent_run_tag g4s-modified-consistent
```

### 8.3. Csak consistency-összehasonlítás egy adott promptszetten

Például a `modified` promptszettre:

```bash
python result_analysis/compare_consistency.py --result_dir evaluation_results --output_dir outputs/pipeline_runs/g4s-modified-consistency-only/report --inconsistent_run_tag g4s-modified-inconsistent --consistent_run_tag g4s-modified-consistent
```

Ugyanez OpenAI-ra:

```bash
python result_analysis/compare_consistency.py --result_dir evaluation_results --output_dir outputs/pipeline_runs/oa-gpt41mini-modified-consistency-only/report --inconsistent_run_tag oa-gpt41mini-modified-inconsistent --consistent_run_tag oa-gpt41mini-modified-consistent
```

Megjegyzés:

- ha `--model_name`-et adsz meg, nem kell a `models.txt`-hez nyúlni
- a `configs/models.txt` átírása helyett inkább külön `configs/models_g4s.txt` fájlt használj
- így a closed-source és open-weight futások modellkészlete nem keveredik el

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

Megjegyzés:

- a `run_pipeline.py` a futás végén automatikusan kiír egy rövid aggregált terminál-összefoglalót is
- ebben provider/model/backend/prompt/consistency bontásban látszik az `overall_acc`, `overall_rmsle` és a `success` arány
- vagyis külön evaluációs parancs nélkül rögtön látod, hogy mennyit talált el a modell
