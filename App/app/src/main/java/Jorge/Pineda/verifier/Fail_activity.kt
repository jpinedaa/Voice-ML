package Jorge.Pineda.verifier

import android.support.v7.app.AppCompatActivity
import android.os.Bundle
import android.widget.TextView

class Fail_activity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_fail_activity)

        val txt = findViewById<TextView>(R.id.distance)
        val intent = getIntent()
        val dist = intent.getDoubleExtra("distance", 0.0)
        txt.setText(dist.toFloat().toString())
    }
}
