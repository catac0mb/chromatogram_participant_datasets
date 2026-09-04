# chromatogram_participant_datasets
This repository contains participant data from studies involving peak detection in chromatograms and human subjects.

## Status 
In-progress analysis of the effect of AI suggestions and suggestion-type on analyst accuracy, engagement, workload, and interaction. Currently includes participant data from two user studies: 1) manual annotation (no_ai) vs annotation with AI peak location suggestions (peaks_only) and 2) annotation with AI peak location suggestions (peaks_only), location suggestions and confidence (confidence), location suggestions and bars indicating features (bars_only), and location suggestions, bars, and confidence (threshold_bars).


## Introduction
This is a project analyzing the effect of AI suggestions and suggestion-type on analyst accuracy, engagement, workload, and interaction. It includes studies with the aforementioned conditions as well as python scripts for analyzing the participant data. Its goal is to examine the impact of AI suggestions and the types of explanations provided with AI suggestions, with conditions increasing in amount of presented information to the user. It is connected to the Peak Annotator repositiory, which is the interface presented as the study.

## Repo Structure
```
├── AI_comparison_participants/     ← participant data from study comparing AI conditions
│   ├── participant_1.json
│   ├── participant_2.json
│   └── ...
│
├── manual_vs_AI_participants/      ← participant data from study comparing manual annotation vs annotation with AI suggestions
│   ├── participant_1.json
│   ├── participant_2.json
│   └── ...
│
├── DATA_DIR/
│   ├── synthetic_data/             ← chromatogram data with 4 different types (see synthetic_chromatogram_datasets repository for generation details)
│   │   ├── control_chromatograms/
│   │   │   ├── annotations/
│   │   │   ├── chromatograms/
│   │   │   └── detections/
│   │   │
│   │   ├── drift_chromatograms/
│   │   │   ├── annotations/
│   │   │   ├── chromatograms/
│   │   │   └── detections/
│   │   │
│   │   ├── noise_chromatograms/
│   │   │   ├── annotations/
│   │   │   ├── chromatograms/
│   │   │   └── detections/
│   │   │
│   │   └── tiny_chromatograms/
│   │       ├── annotations/
│   │       ├── chromatograms/
│   │       └── detections/
│   │
│   └── tuned_parameters_summary.json   ← AI detection parameters (see synthetic_chromatogram_datasets repository for generation details)
│
├── analyze_accuracy.py             ← statistical tests for condition effects on accuracy (per-chromatogram F1, recall, precision, and mean IoU)
├── analyze_edits_vs_accuracy.py    ← statistical tests relating number of annotation edits to accuracy scores
├── analyze_edits_vs_survey.py      ← statistical tests relating number of annotation edits to engagement survey results
├── analyze_edits_vs_tlx.py         ← statistical tests relating number of annotation edits to NASA-TLX workload scores
├── analyze_tlx.py                  ← statistical tests for condition effects on workload
└── factorial_common.py             ← helper functions shared across analysis scripts
```

## Getting Started
### Prerequisites & Needed Materials 
A list of packages and their versions is provided in requirements.txt. To install them, run the following command:

```
pip install -r requirements.txt
```

### Installation 
To install this repo from github, run the following command:
```
git clone https://github.com/washuvis/chromatogram_participant_datasets.git
```

Ensure you have the required packages, as stated in Prerequisites & Needed Materials.

### Usage
There are several python files that take in additional arguments. Examples are provided at the description at the top of each file.

## Future Works
Future work includes running additional studies (different kinds of visualizations, related annotation tasks, etc.) and identifying more measures to include in analyses. Additional measures might include the following: more interactions (mouse hovers, spatial dispersion, etc.), changes in accuracy or user interactions over time, etc. 