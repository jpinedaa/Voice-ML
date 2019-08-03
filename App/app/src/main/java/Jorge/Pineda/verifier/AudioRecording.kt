package Jorge.Pineda.verifier

import android.app.Activity
import android.app.ActivityManager
import android.content.Context
import android.content.Intent
import android.net.Uri
import android.content.res.Resources
import android.graphics.Color
import android.graphics.ColorSpace
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.*
import android.support.v4.content.ContextCompat.startActivity
import android.support.v7.app.AppCompatActivity
import android.util.Log
import android.view.View
import android.widget.ImageButton
import android.widget.TextView
//import com.chaquo.python.Python
import java.nio.ByteBuffer
import org.tensorflow.lite.Interpreter
import java.io.*
import java.nio.ByteOrder
import java.io.File
import android.widget.Toast;

private val AUDIO_SOURCE = MediaRecorder.AudioSource.MIC
private val SAMPLE_RATE = 16000
private val ENCODING = AudioFormat.ENCODING_PCM_16BIT
private val CHANNEL_MASK = AudioFormat.CHANNEL_IN_MONO

private val BUFFER_SIZE = 2 * AudioRecord.getMinBufferSize(
    SAMPLE_RATE,
    CHANNEL_MASK,
    ENCODING
)

private const val T =0.4525525

class AudioRecording(private val mic : ImageButton, private val model: InputStream, private val storeFile: File,
                     private val cont : Activity ,private val enrollment : Boolean, /*private val audi: InputStream,*/
                     private val loading: TextView, private val nstate: Boolean, private val jstate: Boolean, private val filesDir: File): Runnable {

    private var distance: Double = 0.0

    private val handler: Handler = object : Handler(Looper.getMainLooper()) {

        //UI handling
        override fun handleMessage(msg: Message?) {
            Log.d("DEBUG", "its here")
            if (msg != null) {
                when (msg.what) {
                    1 -> mic.setBackgroundColor(Color.LTGRAY)
                    2 -> {
                        val myIntent = Intent(cont, Pass_activity::class.java)
                        myIntent.putExtra("distance", distance)
                        startActivity(cont, myIntent, null)
                    }
                    3 -> {
                        val myIntent = Intent(cont, Fail_activity::class.java)
                        myIntent.putExtra("distance", distance)
                        startActivity(cont, myIntent, null)
                    }
                    in 12..Int.MAX_VALUE -> loading.setText("Processing Audio: " + (msg.what - 12).toString() + "%")
                    5 -> loading.setText("Loading Model")
                    6 -> loading.setText("Inferencing")
                    7 -> loading.setText("Saving")
                    8 -> {
                        loading.setText("Finished")
                        when(done){
                            0->{
                                loading.setText("Your voice is not registered!")
                            }
                            1->{
                                val testIntent = Intent(Intent.ACTION_VIEW, Uri.parse(link1))
                                startActivity(cont, testIntent, null)
                            }
                            2->{
                                val testIntent = Intent(Intent.ACTION_VIEW, Uri.parse(link2))
                                startActivity(cont, testIntent, null)
                            }
                            3->{
                                val testIntent = Intent(Intent.ACTION_VIEW, Uri.parse(link3))
                                startActivity(cont, testIntent, null)
                            }
                        }
                        done=0
                        //Toast.makeText(cont, link1,Toast.LENGTH_LONG).show()

                    }
                    9 -> loading.setText("Opening registered voice data")
                    10 -> loading.setText("Calculating Difference")
                    11 -> {
                        cont.setResult(Activity.RESULT_OK, cont.intent)
                        cont.finish()
                    }
                }
            }
        }
    }

    fun measure_distance(file : File, features : FloatArray) : Double {
        val os = FileInputStream(file).channel
        var filebuffer = ByteBuffer.allocate(features.size * 4)
        os.read(filebuffer)
        filebuffer.flip()
        var floatfilebuffer = filebuffer.asFloatBuffer()
        val storedoutput = FloatArray(features.size)
        floatfilebuffer.get(storedoutput)
        Log.d("output after loading", storedoutput[20].toString())

        handler.obtainMessage(10).apply {
            sendToTarget()
        }

        //val timer11 = SystemClock.elapsedRealtime()
        var dist : Double = 0.0
        for (i in 0 until features.size) {
            dist += (storedoutput[i] - features[i]) * (storedoutput[i] - features[i])
            //distance += (output[0][i] - output[1][i]) * (output[0][i] - output[1][i])
            //Log.d("DEBUG","this 1 " + output[0][i].toString())
            //Log.d("DEBUG","this 2 " + output2[i].toString())
        }
        dist = Math.sqrt(Math.max(dist, 0.0000001))
        return dist

    }

    override fun run() {
        android.os.Process.setThreadPriority(android.os.Process.THREAD_PRIORITY_BACKGROUND)
        //val mic = findViewById<ImageButton>(R.id.mic) as ImageButton
        //while (((mic.background) as ColorDrawable).color != Color.RED) {}

        //recording
        var recorder: AudioRecord? = null
        recorder = AudioRecord(
            AUDIO_SOURCE,
            SAMPLE_RATE,
            CHANNEL_MASK,
            ENCODING,
            BUFFER_SIZE
        )
        recorder.startRecording()

        var buffer = ShortArray(BUFFER_SIZE / 2)
        var audio = ArrayList<Short>()

        //val fileName = File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOCUMENTS), "audiorecord.pcm").path
        //var os = FileOutputStream(fileName)

        var startTime = SystemClock.elapsedRealtime()
        var elapsedTime: Long = 0
        while ((elapsedTime / 1000) < 2) {
            recorder.read(buffer, 0, BUFFER_SIZE / 2)
            for (i in 0 until BUFFER_SIZE / 2) {
                audio.add(buffer[i])
            }
            var temp = SystemClock.elapsedRealtime()
            elapsedTime += (temp - startTime)
            startTime = temp
        }

        recorder?.apply {
            stop()
            release()
        }
        recorder = null

        //os.write(audiobyte, 0, audiobyte.size)

        /*var signal1 = IntArray(audiobyte.size / 2)
        var bb: ByteBuffer = ByteBuffer.wrap(audiobyte)
        var test: Short = 0
        for (i in 0 until 2000) {
            test = bb.getShort()
        }
        signal1[0] = test.toInt();

        Log.d("DEBUG1", signal1[0].toString())
        Log.d("DEBUG1", test.toDouble().toString())

        Log.d("DEBUG1", signal1.size.toString())*/

        var signal = DoubleArray(audio.size)
        for (i in 0 until signal.size) {
            signal[i] = audio[i].toDouble()
        }

        /*
        var bytebuffertest = ByteArrayOutputStream()
        var buffertest = ByteArray(1024)
        var leng = 0
        var total = 0
        while (leng != -1) {
            bytebuffertest.write(buffertest, 0, leng)
            leng = audi.read(buffertest)
            total += leng
        }
        total += 1
        var tobyte = bytebuffertest.toByteArray()
        var bytebuffert = ByteBuffer.wrap(tobyte)
        bytebuffert.order(ByteOrder.nativeOrder())
        var signaltest = DoubleArray((tobyte.size - 44)/2)
        for (i in 0 until 22){
            bytebuffert.getShort()
        }
        for (i in 0 until signaltest.size) {
            signaltest[i] = bytebuffert.getShort().toDouble()
        }*/

        var final1 = Array(100, { Array(40, { FloatArray(3) }) })


        handler.obtainMessage(1).apply {
            sendToTarget()
        }

        //Recording Processing
        if (jstate) {
            Log.d("DEBUGProc", signal.toString())
            var Processing = Processing()
            var processed = Processing.lmfe(signal, 16000, 0.025, 0.01, 40, 400)
            final1 = Processing.extract_derivative_features(processed)

            for (i in 0 until 40) {
                Log.d("DEBUGProc", final1[21][i][0].toString())
                Log.d("DEBUGProc2", final1[21][i][1].toString())
                Log.d("DEBUGProc3", final1[21][i][2].toString())
            }
            Log.d("DEBUGProc", final1[0].size.toString())
        } /*else {

            val py = Python.getInstance()
            val speechpy = py.getModule("speechpy.feature")
            val numpy = py.getModule("numpy")
            val npSignal = numpy.callAttr("array", signal)
            val Processed = speechpy.callAttr("lmfe", npSignal, 16000, 0.025, 0.01, 40, 512)
            val finalProcessed = speechpy.callAttr("extract_derivative_feature", Processed)
            //val tempo = finalProcessed.callAttr("tolist")


            Log.d("PYTHON:", "START")

            /*var temp = arrayOfNulls<List<List<String>>>(100)
        for (i in 0 until 100){
            temp = finalProcessed.asList().toTypedArray()
        }*/


            val temp = finalProcessed.asList()
            for (i in 0 until 100) {
                Log.d("time", i.toString())
                handler.obtainMessage(i + 12).apply {
                    sendToTarget()
                }
                val temp2 = temp.get(i).asList()
                for (j in 0 until 40) {
                    val temp3 = temp2.get(j).asList()
                    for (k in 0 until 3) {
                        final1[i][j][k] = temp3.get(k).toFloat()
                    }
                }
            }

            for (i in 0 until 40) {
                Log.d("PYTHON:", final1[21][i][0].toString())
                Log.d("PYTHON2:", final1[21][i][1].toString())
                Log.d("PYTHON3:", final1[21][i][2].toString())
            }

        }*/
        /*

        var final2 = Array(100, {Array(40, {FloatArray(3)})})

        for (i in 0 until 100){
            for (j in 0 until 40)
            {
                for (k in 0 until 3) {
                    final2[i][j][k] = tempo.asList().get(i + 100).asList().get(j).asList().get(k).toFloat()
                }
            }
        }
*/
        handler.obtainMessage(5).apply {
            sendToTarget()
        }

        var bytebuffer2 = ByteArrayOutputStream()
        var buffer2 = ByteArray(1024)
        var len = 0
        while (len != -1) {
            bytebuffer2.write(buffer2, 0, len)
            len = model.read(buffer2)
        }
        val bytearray = bytebuffer2.toByteArray()
        val modelbuffer = ByteBuffer.allocateDirect(bytearray.size)
        modelbuffer.order(ByteOrder.nativeOrder())
        modelbuffer.put(bytearray)
        var output = arrayOf(FloatArray(1024))
        //var output = arrayOf(FloatArray(1024), FloatArray(1024))
        //val final1 = finalProcessed.copyOfRange(0,100)
        //val final2 = finalProcessed.copyOfRange(100,200)
        //val testfinal = arrayOf(final1,final2)


        //run model and obtain feature vector
        val tfliteOptions = Interpreter.Options().setUseNNAPI(nstate)

        val interpreter = Interpreter(modelbuffer, tfliteOptions)
        val final2 = arrayOf(final1)
        //Log.d("DEBUG",finalProcessed2.size.toString())
        //Log.d("DEBUG",finalProcessed2[0].size.toString())
        //Log.d("DEBUG",finalProcessed2[0][0].size.toString())
        //Log.d("DEBUG",finalProcessed2[0][0][0].size.toString())

        handler.obtainMessage(6).apply {
            sendToTarget()
        }
        val timer1 = SystemClock.elapsedRealtime()
        interpreter.run(final2, output)
        val modeltime = SystemClock.elapsedRealtime() - timer1
        Log.d("Model Time: ", modeltime.toString())

        val output2 = output[0]
        //Log.d("DEBUG","model output " + finalProcessed[0][10][0].toString())
        //Log.d("DEBUG","model output " + processed[0][10].toString())

        if (enrollment) {

            //store feature vector
            Log.d("output before writing", output2[20].toString())
            handler.obtainMessage(7).apply {
                sendToTarget()
            }

            val os = FileOutputStream(storeFile).channel
            var filebuffer = ByteBuffer.allocate(output2.size * 4)
            for (i in 0 until output2.size) {
                filebuffer.putFloat(output2[i])
            }
            filebuffer.flip()
            os.write(filebuffer)

            handler.obtainMessage(11).apply {
                sendToTarget()
            }

        } else {

            //compare stored feature vector with verifying feature vector using euclidean distance
            handler.obtainMessage(9).apply {
                sendToTarget()
            }

            var min : Double = Double.MAX_VALUE
            var user : Int = 0
            var dist: Double = 0.0
            for (i in 1 until 4){
                Log.d("DEBUG","i: " + i.toString())
                val file = File(filesDir, "user" + i.toString() + "Features")
                if (file.isFile){
                    dist = measure_distance(file, output2)
                    if (dist<min){
                        min = dist
                        user = i
                    }
                }
                Log.d("DEBUG","dist: " + dist.toString() + " min: " + min.toString())

            }


            // HERE IS THE RESULT OF DETECTION
            if (min > T){
                //The voice wasnt detected as a registered user
                Log.d("No registered user",min.toString())
                done = 0

            }else {
                //the variable "user" contains the number of the account detected to be the user speaking
                done=user
                Log.d("User detected","user: " + user.toString() + "  distance: " + min.toString() )
            }

            handler.obtainMessage(8).apply {
                sendToTarget()
            }


            /*val os = FileInputStream(storeFile).channel
            var filebuffer = ByteBuffer.allocate(output2.size * 4)
            os.read(filebuffer)
            filebuffer.flip()
            var floatfilebuffer = filebuffer.asFloatBuffer()
            val storedoutput = FloatArray(output2.size)
            floatfilebuffer.get(storedoutput)
            Log.d("output after loading", storedoutput[20].toString())

            handler.obtainMessage(10).apply {
                sendToTarget()
            }

            val timer11 = SystemClock.elapsedRealtime()
            distance = 0.0
            for (i in 0 until output2.size) {
                distance += (storedoutput[i] - output2[i]) * (storedoutput[i] - output2[i])
                //distance += (output[0][i] - output[1][i]) * (output[0][i] - output[1][i])
                //Log.d("DEBUG","this 1 " + output[0][i].toString())
                //Log.d("DEBUG","this 2 " + output2[i].toString())
            }
            distance = Math.sqrt(Math.max(distance, 0.0000001))
            //Log.d("DEBUG","this 3 " + distance.toString())
            var result: Int = 1
            if (distance < T) {
                result = 2
            } else {
                result = 3
            }

            val classtime = SystemClock.elapsedRealtime() - timer11
            Log.d("class Time: ", modeltime.toString())
            handler.obtainMessage(result).apply {
                sendToTarget()
            }*/

        }
        Log.d("DEBUG", "start recordding done")
    }
}
