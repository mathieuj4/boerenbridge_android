package com.boerenbridge.app;

import android.app.Activity;
import android.content.res.AssetManager;
import android.graphics.Color;
import android.os.Bundle;
import android.view.WindowManager;
import android.webkit.WebSettings;
import android.webkit.WebView;
import android.webkit.WebViewClient;

import com.chaquo.python.Python;
import com.chaquo.python.android.AndroidPlatform;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.OutputStream;

public class MainActivity extends Activity {
    private WebView web;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        web = new WebView(this);
        web.setBackgroundColor(Color.parseColor("#14181d"));
        setContentView(web);
        WebSettings s = web.getSettings();
        s.setJavaScriptEnabled(true);
        s.setDomStorageEnabled(true);
        web.setWebViewClient(new WebViewClient());
        web.loadData("<body style='background:#14181d;color:#e8ecf0;font-family:sans-serif;padding:24px'>Starting Boerenbridge...</body>",
                "text/html", "utf-8");
        new Thread(() -> {
            try {
                String dir = copyAssets();
                if (!Python.isStarted()) {
                    Python.start(new AndroidPlatform(this));
                }
                int port = Python.getInstance().getModule("android_main").callAttr("start", dir).toInt();
                runOnUiThread(() -> web.loadUrl("http://127.0.0.1:" + port + "/"));
            } catch (Throwable e) {
                final String msg = android.util.Log.getStackTraceString(e)
                        .replace("&", "&amp;").replace("<", "&lt;");
                runOnUiThread(() -> web.loadData("<body style='background:#14181d;color:#ef7b6e'><pre>" + msg + "</pre></body>",
                        "text/html", "utf-8"));
            }
        }).start();
    }

    private String copyAssets() throws Exception {
        File out = new File(getFilesDir(), "bb");
        out.mkdirs();
        AssetManager am = getAssets();
        String[] names = am.list("bb");
        if (names != null) {
            for (String n : names) {
                try (InputStream in = am.open("bb/" + n); OutputStream o = new FileOutputStream(new File(out, n))) {
                    byte[] buf = new byte[16384];
                    int r;
                    while ((r = in.read(buf)) > 0) o.write(buf, 0, r);
                }
            }
        }
        return out.getAbsolutePath();
    }

    @Override
    public void onBackPressed() {
        moveTaskToBack(true);
    }
}
