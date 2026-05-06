# Futtatási Parancsok

## 1. Gyors lánc-ellenőrzés

Egyetlen, korlátozott konfiguráció fut le, majd automatikusan elkészül a log és a riport.

```bash
python run_pipeline.py --preset quick --model_name gpt5mini
```

Példa szűkített, explicit gyors tesztre:

```bash
python run_pipeline.py --preset quick --model_name gpt5mini --module m0_gravity --equation_difficulty easy --model_system vanilla_equation --law_version v0 --agent_backend vanilla_agent --trials 1 --noise 0.0
```

## 2. Teljes benchmark egy modellel

Az összes modul, difficulty, system és mindkét backend lefut; a `v_unchanged` kontrollverzió alapból ki van zárva, hogy a fő benchmark a 324-es tasktérhez igazodjon.

```bash
python run_pipeline.py --preset benchmark --model_name gpt5mini
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
python run_pipeline.py --preset benchmark --model_name gpt5mini --include_unchanged
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
python run_pipeline.py --preset benchmark --model_name gpt5mini --module m0_gravity --equation_difficulty easy --model_system vanilla_equation --law_version v0 --agent_backend vanilla_agent --prompt_set modified --run_tag v0-compare-inconsistent
```

Utána a konzisztens futás:

```bash
python run_pipeline.py --preset benchmark --model_name gpt5mini --module m0_gravity --equation_difficulty easy --model_system vanilla_equation --law_version v0 --agent_backend vanilla_agent --prompt_set modified --consistency --run_tag v0-compare-consistent
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
python run_pipeline.py --preset benchmark --model_name gpt5mini --include_unchanged --prompt_set original --run_tag all-laws-inconsistent
python run_pipeline.py --preset benchmark --model_name gpt5mini --include_unchanged --prompt_set original --consistency --run_tag all-laws-consistent
```

Az ehhez illeszkedő `modified` páros:

```bash
python run_pipeline.py --preset benchmark --model_name gpt5mini --include_unchanged --prompt_set modified --run_tag all-laws-modified-inconsistent
python run_pipeline.py --preset benchmark --model_name gpt5mini --include_unchanged --prompt_set modified --consistency --run_tag all-laws-modified-consistent
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
python run_pipeline.py --preset benchmark --model_name gpt5mini --api_source oa --include_unchanged --prompt_set original --run_tag oa-gpt5mini-original-inconsistent
python run_pipeline.py --preset benchmark --model_name gpt5mini --api_source oa --include_unchanged --prompt_set original --consistency --run_tag oa-gpt5mini-original-consistent
python run_pipeline.py --preset benchmark --model_name gpt5mini --api_source oa --include_unchanged --prompt_set modified --run_tag oa-gpt5mini-modified-inconsistent
python run_pipeline.py --preset benchmark --model_name gpt5mini --api_source oa --include_unchanged --prompt_set modified --consistency --run_tag oa-gpt5mini-modified-consistent
```

Összesítés:

```bash
python result_analysis/compare_prompt_consistency.py --result_dir evaluation_results --output_dir outputs/pipeline_runs/oa-gpt5mini-prompt-consistency/report --original_inconsistent_run_tag oa-gpt5mini-original-inconsistent --original_consistent_run_tag oa-gpt5mini-original-consistent --modified_inconsistent_run_tag oa-gpt5mini-modified-inconsistent --modified_consistent_run_tag oa-gpt5mini-modified-consistent
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
python result_analysis/compare_consistency.py --result_dir evaluation_results --output_dir outputs/pipeline_runs/oa-gpt5mini-modified-consistency-only/report --inconsistent_run_tag oa-gpt5mini-modified-inconsistent --consistent_run_tag oa-gpt5mini-modified-consistent
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

## 9. Minipaper + reviewer protokoll

Ez az új kísérleti réteg nem a régi benchmark-futtató újabb flag-kombinációja, hanem külön foundation.

Tervezési leírás:

```text
docs/minipaper_reviewer_architecture.md
```

### 9.1. Egyetlen custom scenario

```bash
python scripts/internal/run_minipaper_experiment.py --run_tag minipaper-demo --scientist_model_name gpt5mini --scientist_api_source oa --reviewer_model_name gemma4:31b --reviewer_api_source g4s --modules m0_gravity,m1_coulomb_force --equation_difficulties easy --model_systems vanilla_equation --law_versions v0,v1 --reviewer_can_run_experiments --max_review_rounds 2
```

Ez:

- scientist minipapereket gyártat
- reviewerrel elfogadtatja vagy elutasíttatja őket
- reject esetén maximum `--max_review_rounds` erejéig revise-and-resubmit kört futtat
- és csak az elfogadott papereket teszi a scenario-specifikus knowledge base-be

Kimenet:

```text
outputs/hypothesis_runs/<run_tag>/
```

### 9.2. H1: reviewer tud-e kísérletet futtatni

```bash
python scripts/hypotheses/run_h1_reviewer_experiments.py --scientist_model_name gpt5mini --scientist_api_source oa --max_review_rounds 2
```

Ez két scenariót futtat:

- `reviewer_passive`
- `reviewer_active`

és a végén előállítja:

- `paper_rounds.csv`
- `paper_results.csv`
- `scenario_summary.csv`
- `h1_summary.csv`
- `h1_summary.md`

Teljes G4S többmodelles futás egyetlen shell scriptből:

```bash
bash scripts/hypotheses/H1_runner.sh
```

Ez a `configs/models_g4s.txt` modelljein végigmegy, és a per-run könyvtárak mellett egy közös aggregált könyvtárat is létrehoz:

- `outputs/hypothesis_runs/<run_group_tag>/h1_summary_all.csv`
- `outputs/hypothesis_runs/<run_group_tag>/scenario_summary_all.csv`
- `outputs/hypothesis_runs/<run_group_tag>/paper_results_all.csv`
- `outputs/hypothesis_runs/<run_group_tag>/paper_rounds_all.csv`

Hasznos override-ok shellből:

```bash
RUN_GROUP_TAG=h1-g4s-full SCIENTIST_POPULATION=4 LAW_VERSIONS=v0,v1,v2 bash scripts/hypotheses/H1_runner.sh
```

### 9.3. H2: same-provider vs cross-provider review

```bash
python scripts/hypotheses/run_h2_cross_provider_review.py --openai_model_name gpt5mini --g4s_model_name gemma4:31b --max_review_rounds 2
```

Ez négy scenariót futtat:

- `oa_to_oa`
- `g4s_to_g4s`
- `oa_to_g4s`
- `g4s_to_oa`

és a végén előállítja:

- `paper_rounds.csv`
- `paper_results.csv`
- `scenario_summary.csv`
- `h2_summary.csv`
- `h2_summary.md`

Teljes G4S vs OpenAI többmodelles futás egyetlen shell scriptből:

```bash
bash scripts/hypotheses/H2_runner.sh
```

Ez a `configs/models_g4s.txt` összes modelljére lefuttatja a négy H2 scenariót, és közös aggregált kimeneteket ír:

- `outputs/hypothesis_runs/<run_group_tag>/h2_summary_all.csv`
- `outputs/hypothesis_runs/<run_group_tag>/scenario_summary_all.csv`
- `outputs/hypothesis_runs/<run_group_tag>/paper_results_all.csv`
- `outputs/hypothesis_runs/<run_group_tag>/paper_rounds_all.csv`

Ha a reviewernek is engedélyezni akarod a kísérletezést:

```bash
REVIEWER_CAN_RUN_EXPERIMENTS=1 bash scripts/hypotheses/H2_runner.sh
```

### 9.4. H1 + H2 teljes G4S batch egyetlen paranccsal

```bash
bash scripts/hypotheses/run_all_hypotheses_g4s.sh
```

Ez egymás után elindítja:

- `scripts/hypotheses/H1_runner.sh`
- `scripts/hypotheses/H2_runner.sh`

és külön aggregate directory-kat hoz létre H1-re és H2-re.

Megjegyzés:

- a minipaper protokoll mindig a konzisztens fizikai változtatásokat használja
- a shared context csak az elfogadott minipaperekből épül
- ez lényegesen drágább API-hívásszámú workflow, mint a sima benchmark runner
