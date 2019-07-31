package Jorge.Pineda.verifier

import android.app.Activity
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.content.res.Resources
import android.graphics.Color
import android.graphics.drawable.ColorDrawable
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaPlayer
import android.media.MediaRecorder
import android.os.*
import android.support.v7.app.AppCompatActivity
import android.os.SystemClock.elapsedRealtime
import android.support.v4.app.ActivityCompat
import android.support.v4.content.ContextCompat
import android.util.Log
import android.view.View
import android.widget.ImageButton
import android.widget.TextView
import java.io.File
import java.io.IOException

private const val LOG_TAG = "AudioRecordTest"

class Enrollment_activity : AppCompatActivity() {

    private var fileName: String = ""

    private var handler: Handler?= null

    //private var recorder: MediaRecorder? = null
    private var player: MediaPlayer? = null

    private var recorder: AudioRecord? = null

    var mStartRecording = true

    private var AUDIO_SOURCE = MediaRecorder.AudioSource.MIC
    private var SAMPLE_RATE = 16000
    private var ENCODING = AudioFormat.ENCODING_PCM_16BIT
    private var CHANNEL_MASK = AudioFormat.CHANNEL_IN_MONO

    private var BUFFER_SIZE = 2 * AudioRecord.getMinBufferSize(SAMPLE_RATE, CHANNEL_MASK, ENCODING)

    private fun onRecord(start: Boolean) = if (start) {
        startRecording()
    } else {
        stopRecording()
    }

    private fun onPlay(start: Boolean) = if (start) {
        startPlaying()
    } else {
        stopPlaying()
    }

    private fun startPlaying() {
        player = MediaPlayer().apply {
            try {
                setDataSource(fileName)
                prepare()
                start()
            } catch (e: IOException) {
                Log.e(LOG_TAG, "prepare() failed")
            }
        }
    }

    private fun stopPlaying() {
        player?.release()
        player = null
    }

   /* private fun startRecording() {
        recorder = MediaRecorder().apply {
            setAudioSource(MediaRecorder.AudioSource.MIC)
            setOutputFormat(MediaRecorder.OutputFormat.MPEG_4)
            setOutputFile(fileName)
            setAudioChannels(1)
            setAudioEncoder(/*MediaRecorder.AudioEncoder.AAC*/AudioFormat.ENCODING_PCM_16BIT)
            setAudioSamplingRate(16000)

            try {
                prepare()
            } catch (e: IOException) {
                Log.e(LOG_TAG, "prepare() failed")
            }

            start()
        }
    }*/

    private fun startRecording() {
        val mic = findViewById<ImageButton>(R.id.mic) as ImageButton
        while (((mic.background) as ColorDrawable).color != Color.RED) {

        }


        recorder = AudioRecord(AUDIO_SOURCE ,SAMPLE_RATE , CHANNEL_MASK, ENCODING, BUFFER_SIZE)
        recorder?.startRecording()

        var buffer = ByteArray(BUFFER_SIZE)

        //var os = FileOutputStream(fileName)

        var startTime = elapsedRealtime()
        var elapsedTime: Long = 0

        while((elapsedTime/1000)< 3) {
            recorder?.read(buffer, 0, BUFFER_SIZE)
            //os.write(buffer, 0, BUFFER_SIZE)
            var temp = elapsedRealtime()
            elapsedTime += (temp - startTime)
            startTime = temp
        }
        stopRecording()



   }

    private fun stopRecording() {
        recorder?.apply {
            stop()
            release()
        }
        recorder = null
    }


    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.enrollment_view)

        //fileName = "${externalCacheDir.absolutePath}/audiorecord.pcm"

        //ActivityCompat.requestPermissions(this, arrayOf(android.Manifest.permission.WRITE_EXTERNAL_STORAGE), 300)

        //val fileName2 = File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOCUMENTS), "audiorecord.pcm")
        //fileName = fileName2.path


        if (ContextCompat.checkSelfPermission(this , android.Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {

        } else {

            val mic = findViewById<ImageButton>(R.id.mic)
            mic.setBackgroundColor(Color.LTGRAY)
            /*mic.setOnClickListener {
                Log.d("DEBUG", "before color change")
                mic.setBackgroundColor(Color.RED)
                debug_text.setText("mstartrecording = " + mStartRecording)
                Log.d("DEBUG","b4 start recordding done")
                startRecording()
                Log.d("DEBUG","start recordding done")
                //if (mStartRecording) {
                  //  mic.setBackgroundColor(Color.RED)
                //} else {
                 //   mic.setBackgroundColor(Color.LTGRAY)
               // }
                //onRecord(mStartRecording, mic)
                //onPlay(!mStartRecording)
                mStartRecording = !mStartRecording
            }*/

        }

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

        val mic = findViewById<ImageButton>(R.id.mic)

        //converted tensorflow lite model
        val model = resources.openRawResource(R.raw.voicenet)

        //val audio = resources.openRawResource(R.raw.testaudio)

        //file where enrolled voice vector is stored
        val storeFile = File(filesDir, "userFeatures")

        val loading = findViewById<TextView>(R.id.loading)
        //val nstate = getIntent().getBooleanExtra("nnapi", false)
        //val jstate = getIntent().getBooleanExtra("java", false)

        val record = AudioRecording(mic, model, storeFile, this ,true ,/*audio,*/ loading, true, true, filesDir)
        //val record_thread = Thread(record)
        //record_thread.start()
        //startRecording()

        Thread(record).start()

        Log.d("DEBUG","fininshed llistener")
        //if (mStartRecording) {
        //  mic.setBackgroundColor(Color.RED)
        //} else {
        //   mic.setBackgroundColor(Color.LTGRAY)
        // }
        //onRecord(mStartRecording, mic)
        //onPlay(!mStartRecording)
        //mStartRecording = !mStartRecording


    }
    override fun onStop() {
        super.onStop()
        recorder?.release()
        recorder = null
        player?.release()
        player = null
    }
}


