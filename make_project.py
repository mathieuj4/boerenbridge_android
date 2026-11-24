"""Builds the Android Studio/Gradle project (folders and all) from the flat files next to this script."""
import os, shutil
here = os.path.dirname(os.path.abspath(__file__))
root = os.path.join(here, "android")
shutil.rmtree(root, ignore_errors=True)
def put(src, dst):
    p = os.path.join(root, dst); os.makedirs(os.path.dirname(p), exist_ok=True); shutil.copy(os.path.join(here, src), p)
put("settings.gradle.kts", "settings.gradle.kts")
put("root_build.gradle.kts", "build.gradle.kts")
put("app_build.gradle.kts", "app/build.gradle.kts")
put("AndroidManifest.xml", "app/src/main/AndroidManifest.xml")
put("MainActivity.java", "app/src/main/java/com/boerenbridge/app/MainActivity.java")
put("ic_launcher.png", "app/src/main/res/mipmap-xxxhdpi/ic_launcher.png")
for f in ["boerenbridge", "agents", "agents2", "opponents", "ismcts", "tuned", "fast", "planner", "sim_agent",
          "valuenet", "bidnet", "nn", "play", "android_main"]:
    put(f + ".py", f"app/src/main/python/{f}.py")
for f in ["index.html", "bidnet.npz", "valuenet.npz", "tuned_theta.npy"]:
    put(f, f"app/src/main/assets/bb/{f}")
print("project written to", root)
