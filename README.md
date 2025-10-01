python package
```
pip install opencv-python
pip install numpy
pip install pandas
pip install python-dotenv
pip install mediapipe
pip install torch torchvision
pip install requests
```

Gemini API
```
pip install google-generativeai
```

Directory Structure
```project/
├── video/
│   └── r1test.mp4          # 분석할 비디오 파일
|   └── r1test_gt.csv       # 분석할 GT 파일
├── results/                # 자동 생성
|   └── r1test_vlm_20251001_154428.json
|   └── results/r1test_vlm_20251001_154427.csv
├── robot_scripts/          # 자동 생성
|   └── r1test_robot_script_20251001_154428.py
├── logs/                   # 자동 생성
├── temp/                   # 자동 생성
├── accuracy_results/       # 자동 생성
├── success_main.py         # 첫 번째 스크립트
├── success_analysis.py     # 두 번째 스크립트
└── .env                    # API 키 저장
```

