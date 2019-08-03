package Jorge.Pineda.verifier

import android.app.Activity
import android.content.Intent
import android.net.Uri
import android.support.v7.app.AppCompatActivity
import android.os.Bundle
import android.util.Log
import android.widget.Button
import android.widget.EditText
import android.widget.TextView
import android.widget.Toast
import java.io.File
import android.util.TypedValue



class Account_setting : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.account_setting)

        val id = findViewById<EditText>(R.id.account_id)

        if (intent.getStringExtra("id")!="") {
            id.setText(intent.getStringExtra("id"))
        }
        val getlink = findViewById<EditText>(R.id.link)

        getlink.setText(intent.getStringExtra("link"))

        val register_status = findViewById<TextView>(R.id.register_status)
        val to = File(filesDir, "user" + intent.getIntExtra("account_no", 0).toString() + "Features")
        if (to.isFile){
            register_status.setText("VoiceID registered")
        } else register_status.setText("Voice not registered yet!")

        val intentR = Intent()
        val save = findViewById<Button>(R.id.save_button)
        save.setOnClickListener{
            intentR.putExtra("link",getlink.text.toString())
            intentR.putExtra("id",id.text.toString())
            setResult(Activity.RESULT_OK,intentR)
            finish()
        }

        //go to enrollment activity
        val enroll = findViewById<Button>(R.id.button_register)
        enroll.setOnClickListener{
            val myIntent = Intent(this, Enrollment_activity::class.java)
            startActivityForResult(myIntent,1)
        }
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        Log.d("DEBUG","returned register activity2")
        if(resultCode == Activity.RESULT_OK){
            Log.d("DEBUG","returned register activity3")
            val from = File(filesDir, "userFeatures")
            val to = File(filesDir, "user" + intent.getIntExtra("account_no", 0).toString() + "Features")
            val register_status = findViewById<TextView>(R.id.register_status)
            if(to.isFile){
                to.delete()
            }
            when(requestCode){
                1 -> {
                    Log.d("DEBUG","returned register activity")
                    from.renameTo(to)
                    register_status.setText("VoiceID registered")
                }
            }
        }
    }
}