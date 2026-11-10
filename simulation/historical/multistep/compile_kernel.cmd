@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1
cl /nologo /O2 /fp:precise /EHsc /std:c++17 /LD "D:\FinsSim-deps\Unity\de101-multistep-bias-200-20261109\pkg\finshybrid\native\multistep_bias_kernel.cpp" /link /EXPORT:multistep_bias_rollout /OUT:multistep.dll
