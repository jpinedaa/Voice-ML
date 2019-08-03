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
import org.json.JSONArray
import org.json.JSONObject


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
        val features_account = File(filesDir, "user" + intent.getIntExtra("account_no", 0).toString() + "Features")
        if (features_account.isFile){
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

        val delete_id = findViewById<Button>(R.id.delete_button)
        delete_id.setOnClickListener{
            if (features_account.isFile){
                features_account.delete()
            }

            val ids_data = File(filesDir, "ids_data")
            val ids_obj = JSONObject(ids_data.readText())
            var ids_array  = ids_obj.get("users")
            ids_array = ids_array as JSONArray
            val user = JSONObject()
            user.put("id", 0)
            user.put("link", 0)
            ids_array.put(intent.getIntExtra("account_no", 3) - 1 , user)
            val new_json = JSONObject()
            new_json.put("users", ids_array)
            ids_data.delete()
            ids_data.writeText(new_json.toString())

            val IntentR = Intent()
            IntentR.putExtra("link", "")
            IntentR.putExtra("ids", "")

            register_status.setText("Voice not registered yet!")
            id.setText("")
            getlink.setText("")
            setResult(Activity.RESULT_OK,intentR)
            finish()

        }
    }

    public override fun onResume() {
        super.onResume()
        // put your code here...
        val id = findViewById<EditText>(R.id.account_id)
        val getlink = findViewById<EditText>(R.id.link)
        id.setText(intent.getStringExtra("id"))
        getlink.setText(intent.getStringExtra("link"))

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