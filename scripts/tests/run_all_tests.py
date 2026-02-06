# scripts/tests/run_all_tests.py
import subprocess
import sys
import os
import time
import logging

def configure_test_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', force=True)

def run_command(command, description, stop_on_fail=True):
    print(f"\n>>> Running: {description} ...")
    print(f"    Command: {command}")
    start_time = time.time()
    
    # 環境変数の設定 (重要: PYTHONPATHを通す)
    env = os.environ.copy()
    env["SNN_TEST_MODE"] = "1"
    env["PYTHONWARNINGS"] = "ignore"
    
    # プロジェクトルートをPYTHONPATHに追加
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    current_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{project_root}:{current_pythonpath}"
    
    noise_filters = [
        "No module named 'cupy'",
        "spikingjelly",
        "Matplotlib is building the font cache",
        "percent"
    ]

    process = subprocess.Popen(
        command, 
        shell=True, 
        env=env,
        cwd=project_root, # カレントディレクトリも明示
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True
    )

    if process.stdout:
        for line in process.stdout:
            if any(noise in line for noise in noise_filters):
                continue
            print(line, end='')
    
    process.wait()
    duration = time.time() - start_time
    
    if process.returncode == 0:
        print(f"✅ {description} Passed ({duration:.2f}s)")
        return True
    else:
        print(f"❌ {description} Failed (Exit Code: {process.returncode})")
        if stop_on_fail:
            return False
        return False

def main():
    configure_test_logging()
    
    print("==================================================")
    print("   DORA Neuromorphic OS - Full System Validation  ")
    print("   Target: Phase 2 (Practical Learning Capability)")
    print("==================================================")
    
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    os.chdir(project_root)
    
    all_tests_passed = True

    # 1. System Health Check
    if not run_command("python scripts/tests/run_project_health_check.py", "1. Project Health Check"):
        sys.exit(1)

    # 2. Unit Tests
    print("\n--- 2. Unit Tests (Core Logic) ---")
    if not run_command("python -m pytest tests/ -v -s", "Core Unit Tests", stop_on_fail=False):
        all_tests_passed = False

    # 3. Learning Function Tests (Recipes)
    print("\n--- 3. Learning Capability Tests (Recipes) ---")
    recipes = [
        ("snn_research/recipes/mnist.py", "MNIST Learning Recipe"),
    ]
    for script, desc in recipes:
        if os.path.exists(script):
            # モジュールとして実行することでインポートエラーを回避
            module_name = script.replace("/", ".").replace(".py", "")
            if not run_command(f"python -m {module_name} --epochs 1", desc, stop_on_fail=False):
                all_tests_passed = False

    # 4. Brain Experiment Tests
    print("\n--- 4. Brain Experiment Tests ---")
    experiments = [
        "scripts/experiments/brain/run_phase2_mnist_tuning.py",
    ]
    for script in experiments:
        if os.path.exists(script):
            if not run_command(f"python {script}", f"Experiment: {os.path.basename(script)}", stop_on_fail=False):
                all_tests_passed = False

    # 5. Benchmarks & Verification
    print("\n--- 5. Benchmarks & Verification ---")
    verification_scripts = [
        "scripts/tests/run_compiler_test.py",
        "scripts/tests/verify_phase3.py",
        "scripts/tests/verify_performance.py",
    ]
    for script in verification_scripts:
        if os.path.exists(script):
            if not run_command(f"python {script}", f"Verify: {os.path.basename(script)}", stop_on_fail=False):
                all_tests_passed = False

    print("\n==================================================")
    if all_tests_passed:
        print("🎉 ALL SYSTEMS GO: Ready for Practical Deployment.")
        sys.exit(0)
    else:
        print("⚠️ SYSTEM UNSTABLE: Please fix the failed modules above.")
        sys.exit(1)

if __name__ == "__main__":
    main()