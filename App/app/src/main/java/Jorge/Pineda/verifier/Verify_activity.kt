package Jorge.Pineda.verifier

import android.graphics.Color
import android.support.v7.app.AppCompatActivity
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.Button
import android.widget.ImageButton
import android.widget.TextView
import java.io.File
import android.content.Intent
import android.net.Uri
import android.widget.Toast

var done:Int=0
class Verify_activity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.verify_view)

        val mic = findViewById<ImageButton>(R.id.mic2)
        mic.setBackgroundColor(Color.LTGRAY)

        /*val message = intent.getStringExtra(EXTRA_MESSAGE)

        val textView = findViewById<TextView>(R.id.textView).apply {
            text = message
        }*/
    }

    fun micOnClickListener(view: View) {
        Log.d("DEBUG", "before color change")

        view.setBackgroundColor(Color.RED)

        //val debug_text = findViewById<TextView>(R.id.debug_text)
        //debug_text.setText("mstartrecording = " + mStartRecording)
        Log.d("DEBUG","b4 start recordding done")

        val mic = findViewById<ImageButton>(R.id.mic2)

        //converted tensorflow lite model
        val model = resources.openRawResource(R.raw.voicenet)

        //val audi = resources.openRawResource(R.raw.testaudio)

        //file where enrolled voice vector is stored
        val storeFile = File(filesDir, "userFeatures")
        val loading = findViewById<TextView>(R.id.loading)
        val nstate = getIntent().getBooleanExtra("nnapi", false)
        val jstate = getIntent().getBooleanExtra("java", false)


        val record = AudioRecording(mic, model, storeFile, this,false,/*audi,*/ loading, nstate, jstate,filesDir)
        val t = Thread(record)
        t.start()
        t.join()
        while (done==0){}
        when(done){
            1->{
                val testIntent = Intent(Intent.ACTION_VIEW, Uri.parse(link1))
                startActivity(testIntent)
            }
            2->{
                val testIntent = Intent(Intent.ACTION_VIEW, Uri.parse(link2))
                startActivity(testIntent)
            }
            3->{
                val testIntent = Intent(Intent.ACTION_VIEW, Uri.parse(link3))
                startActivity(testIntent)
            }
        }
        done=0
        Toast.makeText(this, link1,Toast.LENGTH_LONG).show()
        Log.d("DEBUG","fininshed llistener")

    }
}
