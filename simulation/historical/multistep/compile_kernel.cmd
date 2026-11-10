@echo off
if not defined PI_UV_VCVARS64 (echo Set PI_UV_VCVARS64 to vcvars64.bat & exit /b 1)
if not defined PI_UV_KERNEL_SOURCE (echo Set PI_UV_KERNEL_SOURCE to multistep_bias_kernel.cpp & exit /b 1)
call "%PI_UV_VCVARS64%" >nul || exit /b 1
cl /nologo /O2 /fp:precise /EHsc /std:c++17 /LD "%PI_UV_KERNEL_SOURCE%" /link /EXPORT:multistep_bias_rollout /OUT:multistep.dll
