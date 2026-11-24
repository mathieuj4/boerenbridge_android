plugins {
    id("com.android.application")
    id("com.chaquo.python")
}

android {
    namespace = "com.boerenbridge.app"
    compileSdk = 34
    defaultConfig {
        applicationId = "com.boerenbridge.app"
        minSdk = 24
        targetSdk = 34
        versionCode = 1
        versionName = "1.0"
        ndk {
            abiFilters += listOf("arm64-v8a", "x86_64")
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
}

chaquopy {
    defaultConfig {
        version = "3.12"
        pip {
            install("numpy")
        }
    }
}
