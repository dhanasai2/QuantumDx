@echo off
cd /d D:\QuantumDx
python -W ignore -u -m src.large_dataset.adaptive_qsvm_experiment > adaptive_qsvm_run.log 2> adaptive_qsvm_run.err.log
