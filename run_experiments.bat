@echo off
setlocal enabledelayedexpansion
chcp 65001 > nul

echo ============================================================
echo  Fraud Detection GNN - Elliptic Dataset Experiment Suite
echo ============================================================
echo.

:: ============================================================
:: Check Python + CUDA availability
:: ============================================================
python -c "import torch; print('PyTorch:', torch.__version__, '| CUDA:', torch.cuda.is_available(), '| GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Python or PyTorch not available!
    pause & exit /b 1
)
echo.

:: ============================================================
:: SECTION 1: Traditional GNN Baselines (Naive - Rerun)
::   Reason: Old format CSVs or wrong EWC config
:: ============================================================
echo [SECTION 1] Traditional GNN Baselines (Elliptic Naive)
echo --------------------------------------------------------

echo [1/6] GCN Naive...
python train.py --config configs/traditional/GCN/elliptic_Naive_GCN.yaml
if %ERRORLEVEL% neq 0 ( echo [WARN] GCN Naive failed, continuing... )

echo [2/6] GAT Naive...
python train.py --config configs/traditional/GAT/elliptic_Naive_GAT.yaml
if %ERRORLEVEL% neq 0 ( echo [WARN] GAT Naive failed, continuing... )

echo [3/6] GIN Naive...
python train.py --config configs/traditional/GIN/elliptic_Naive_GIN.yaml
if %ERRORLEVEL% neq 0 ( echo [WARN] GIN Naive failed, continuing... )

echo [4/6] GATv2 Naive...
python train.py --config configs/traditional/GATv2/elliptic_Naive_GATv2.yaml
if %ERRORLEVEL% neq 0 ( echo [WARN] GATv2 Naive failed, continuing... )

echo [5/6] GraphSAGE Naive...
python train.py --config configs/traditional/GraphSage/elliptic_Naive_GraphSage.yaml
if %ERRORLEVEL% neq 0 ( echo [WARN] GraphSAGE Naive failed, continuing... )

echo [6/6] GraphSMOTE Naive...
python train.py --config configs/traditional/GraphSMOTE/elliptic_Naive_GraphSMOTE.yaml
if %ERRORLEVEL% neq 0 ( echo [WARN] GraphSMOTE Naive failed, continuing... )

echo.

:: ============================================================
:: SECTION 2: GCN CL Baselines (New experiments)
:: ============================================================
echo [SECTION 2] GCN Continual Learning Baselines
echo --------------------------------------------------------

echo [1/3] GCN + EWC...
python train.py --config configs/traditional/GCN/elliptic_EWC_GCN.yaml
if %ERRORLEVEL% neq 0 ( echo [WARN] EWC-GCN failed, continuing... )

echo [2/3] GCN + ER (Experience Replay)...
python train.py --config configs/traditional/GCN/elliptic_CL_GCN.yaml
if %ERRORLEVEL% neq 0 ( echo [WARN] ER-GCN failed, continuing... )

echo [3/3] GCN + LwF...
python train.py --config configs/traditional/GCN/elliptic_LwF_GCN.yaml
if %ERRORLEVEL% neq 0 ( echo [WARN] LwF-GCN failed, continuing... )

echo.

:: ============================================================
:: SECTION 3: BSL CL Baselines (for comparison with TASD-CL)
:: ============================================================
echo [SECTION 3] BSL Continual Learning Baselines
echo --------------------------------------------------------

echo [1/3] BSL + EWC...
python train.py --config configs/BSL/elliptic_EWC_BSL.yaml
if %ERRORLEVEL% neq 0 ( echo [WARN] EWC-BSL failed, continuing... )

echo [2/3] BSL + ER...
python train.py --config configs/BSL/elliptic_ER_BSL.yaml
if %ERRORLEVEL% neq 0 ( echo [WARN] ER-BSL failed, continuing... )

echo [3/3] BSL + LwF...
python train.py --config configs/BSL/elliptic_LwF_BSL.yaml
if %ERRORLEVEL% neq 0 ( echo [WARN] LwF-BSL failed, continuing... )

echo.

:: ============================================================
:: SECTION 4: Our Method - TASD-CL
:: ============================================================
echo [SECTION 4] TASD-CL (Ours)
echo --------------------------------------------------------

echo [1/1] TASD-CL on BSL...
python train.py --config configs/BSL/elliptic_TASDCL_BSL.yaml
if %ERRORLEVEL% neq 0 ( echo [WARN] TASD-CL BSL failed, continuing... )

echo.

:: ============================================================
:: SECTION 5: Ablation Study
:: ============================================================
echo [SECTION 5] Ablation Study
echo --------------------------------------------------------

echo [1/3] TASD-CL w/o SSF (SPC+SCD only)...
python train.py --config configs/BSL/elliptic_TASDCL_noSSF_BSL.yaml
if %ERRORLEVEL% neq 0 ( echo [WARN] TASD-CL noSSF failed, continuing... )

echo [2/3] TASD-CL w/o SPC (SSF+SCD only)...
python train.py --config configs/BSL/elliptic_TASDCL_noSPC_BSL.yaml
if %ERRORLEVEL% neq 0 ( echo [WARN] TASD-CL noSPC failed, continuing... )

echo [3/3] TASD-CL w/o SCD (SSF+SPC only)...
python train.py --config configs/BSL/elliptic_TASDCL_noSCD_BSL.yaml
if %ERRORLEVEL% neq 0 ( echo [WARN] TASD-CL noSCD failed, continuing... )

echo.

:: ============================================================
:: SECTION 6: Analysis
:: ============================================================
echo [SECTION 6] Generating Analysis Report
echo --------------------------------------------------------
python analyze_results.py
if %ERRORLEVEL% neq 0 ( echo [WARN] Analysis script failed. )

echo.
echo ============================================================
echo  All experiments completed!
echo  Results saved to: results_summary.csv
echo  Report saved to:  results_report.txt
echo ============================================================
pause
