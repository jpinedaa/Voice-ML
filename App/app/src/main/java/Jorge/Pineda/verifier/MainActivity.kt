package Jorge.Pineda.verifier

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.support.v7.app.AppCompatActivity
import android.os.Bundle
import android.support.v4.app.ActivityCompat.*
import android.widget.Button
import android.widget.Switch

private const val REQUEST_RECORD_AUDIO_PERMISSION = 200

class MainActivity : AppCompatActivity() {

    private var permissionToRecordAccepted = false
    private var permissions: Array<String> = arrayOf(Manifest.permission.RECORD_AUDIO)

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        permissionToRecordAccepted = if (requestCode == REQUEST_RECORD_AUDIO_PERMISSION) {
            grantResults[0] == PackageManager.PERMISSION_GRANTED
        } else {
            false
        }
        if (!permissionToRecordAccepted) finish()
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        requestPermissions(this, permissions, REQUEST_RECORD_AUDIO_PERMISSION)

        var nstate = false
        var jstate = false
        val nnapi = findViewById<Switch>(R.id.nnapi)
        val java = findViewById<Switch>(R.id.java)
        nnapi.setOnCheckedChangeListener {_, isChecked ->
            nstate = isChecked
        }
        java.setOnCheckedChangeListener {_, isChecked ->
            jstate = isChecked
        }

        val enroll = findViewById<Button>(R.id.enroll)
        enroll.setOnClickListener{
            val myIntent = Intent(this, Enrollment_activity::class.java)
            myIntent.putExtra("nnapi", nstate)
            myIntent.putExtra("java", jstate)
            startActivity(myIntent)
        }

        val verify = findViewById<Button>(R.id.verify)
        verify.setOnClickListener{
            val myIntent2 = Intent(this, Verify_activity::class.java)
            myIntent2.putExtra("nnapi", nstate)
            myIntent2.putExtra("java", jstate)
            startActivity(myIntent2)
        }
    }


/*
    fun sendMessage(view: View) {
        val editText = findViewById<EditText>(R.id.editText)
        val message = editText.text.toString()
        val intent = Intent(this, DisplayMessageActivity::class.java).apply {
            putExtra(EXTRA_MESSAGE, message)
        }
        startActivity(intent)
    }*/
}


