import subprocess

# Run pytest using the current venv's python and capture to text file
result = subprocess.run(
    ["venv\\Scripts\\python.exe", "-m", "pytest", "tests/", "-v"],
    capture_output=True,
    text=True,
    encoding="utf-8",
    errors="replace"
)

with open("test_traceback.txt", "w", encoding="utf-8") as f:
    f.write(result.stdout)
    f.write("\n\nSTDERR:\n")
    f.write(result.stderr)

print("Done writing test_traceback.txt")
