from finshybrid.experiment_paths import config_path, load_record, relocate_text
import csv
import shutil
import subprocess
from paths import BASE, EMPTY, LOADED, PILOT, WORK, ROOT, PAIRS, read, save, check, sha, verify

def substitute(source, old, new):
    check(source.count(old) == 1, old)
    return source.replace(old, new)

def build():
    check(not (ROOT / "source_freeze.json").exists(), "already frozen")
    src = ROOT / "pkg/finshybrid"
    shutil.copytree(WORK / "src/finshybrid", src, ignore=shutil.ignore_patterns("__pycache__", "assets"), dirs_exist_ok=True)
    m = src / "multistep_bias.py"
    s = m.read_text(encoding="utf-8")
    # Preserve the rollout and estimator verbatim; adapt only the library loader to Windows.
    begin = s.index('        source = Path(__file__)')
    end = s.index('        self.library =', begin)
    s = s[:begin] + '        library_path = Path(cache_dir) / "multistep.dll"\n' + s[end:]
    m.write_text(s, encoding="utf-8")
    compiler = config_path("vcvars64")
    script = ROOT / "compile_kernel.cmd"
    script.write_text(f'@echo off\ncall "{compiler}" >nul || exit /b 1\ncl /nologo /O2 /fp:precise /EHsc /std:c++17 /LD "{src / "native/multistep_bias_kernel.cpp"}" /link /EXPORT:multistep_bias_rollout /OUT:multistep.dll\n',encoding="utf-8")
    subprocess.run(["cmd", "/d", "/c", str(script)], cwd=ROOT, check=True)
    editor = config_path("unity_editor_data")
    for load, original in [("empty", BASE), ("ballast300", PILOT)]:
        dest = ROOT / load
        shutil.copytree(original / "player", dest / "player", dirs_exist_ok=True)
        smoke = "FinsRovHybridBasisSmoke.cs"
        original_source = BASE / "unity_source/Assets/Scripts" / smoke if load=="empty" else PILOT / smoke
        s = original_source.read_text(encoding="utf-8-sig")
        s = substitute(s, "  De92Reset();de85Variance", "  ResetMultistep();De92Reset();de85Variance")
        s = substitute(s, 'if(De83Online=="de92")av=De92Update(t,ev,de83Previous,out vb,out va);',
                       'if(De83Online=="de92")av=MultistepUpdate(t,de83Previous,u,current,out vb,out va);')
        (dest / smoke).write_text(s, encoding="utf-8")
        dll = dest / "player/FinsROVHybridBasis_Data/Managed/Assembly-CSharp.dll"
        lines = relocate_text((BASE / "compile/Assembly-CSharp.player.rsp").read_text(encoding="utf-8-sig")).splitlines()
        lines[0] = f'-out:"{dll}"'
        lines[lines.index(f'"Assets/Scripts/{smoke}"')] = f'"{dest/smoke}"'
        lines.append(f'"{ROOT/"SimBias.cs"}"')
        if load=="ballast300":
            lines.append(f'"{PILOT/"Ballast.cs"}"')
        rsp = dest/"compile.rsp"
        rsp.write_text("\n".join(lines),encoding="utf-8")
        subprocess.run([editor/"NetCoreRuntime/dotnet.exe",editor/"DotNetSdkRoslyn/csc.dll","@"+str(rsp)],cwd=BASE/"unity_source",check=True)

def plan():
    with PAIRS.open(encoding="utf-8-sig",newline="") as stream:
        rows = list(csv.DictReader(stream))
    old = read(EMPTY/"plan400.json")
    loaded = read(LOADED/"plan400.json")
    check(len(old)==len(loaded)==400 and len(rows)==100,"plan counts")
    jobs = []
    files = {PAIRS, EMPTY/"plan400.json", LOADED/"plan400.json", WORK/"src/finshybrid/multistep_bias.py"}
    for i, (a,b) in enumerate(zip(old,loaded)):
        if "/H_bias/" not in a["id"]:
            continue
        check((a["group"],a["id"])==(b["group"],b["id"]), "plan mismatch")
        g = a["group"]
        c, init, arm, seed = a["id"].split("/")
        match = [r for r in rows if (r["group"],r["environment"],r["seed"])==(g,c,seed[4:])]
        check(len(match)==1,"missing paired condition")
        settings = [{r["key"]:r["value"] for r in j["environment"]} for j in (a,b)]
        check({k:v for k,v in settings[0].items() if k!="FINSSIM_EDMD_MPC_DIR"}==
              {k:v for k,v in settings[1].items() if k!="FINSSIM_EDMD_MPC_DIR"},"non-physical mismatch")
        check(settings[0]["FINSSIM_EDMD_MPC_MODEL_PATH"]==match[0]["model"],"model")
        for k in ("FINSSIM_EDMD_MPC_MODEL_PATH","FINSSIM_EDMD_THRUSTER_CURVE_PATH","FINSSIM_DE90_FLOW_FILE"):
            files.add(Path(settings[0][k]))
        for load,env in zip(("empty","ballast300"),settings):
            parent = env["FINSSIM_EDMD_MPC_DIR"]
            check(parent==match[0]["old_empty" if load=="empty" else "old_ballast"],"baseline path")
            output = ROOT/load/g/"formal"/a["id"]
            env["FINSSIM_EDMD_MPC_DIR"] = str(output)
            jobs.append(dict(load=load,group=g,id=a["id"],old_index=i%40,old_folder=parent,env=env))
    check(len({(j["load"],j["group"],j["id"]) for j in jobs})==200,"200 unique jobs")
    save(ROOT/"plan200.json",jobs)
    for pattern in ("*.py","*.cs","*.cmd","multistep.dll","plan200.json","pkg/**/*.py","pkg/**/*.cpp",
                    "*/FinsRovHybridBasisSmoke.cs","*/compile.rsp","*/player/FinsROVHybridBasis_Data/Managed/Assembly-CSharp.dll"):
        files.update(ROOT.glob(pattern))
    files.add(PILOT/"Ballast.cs")
    save(ROOT/"source_freeze.json",{str(f):sha(f) for f in sorted(files)})

from pathlib import Path

if __name__=="__main__":
    build()
    plan()
