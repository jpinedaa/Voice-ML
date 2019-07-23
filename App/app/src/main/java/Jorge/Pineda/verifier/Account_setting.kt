package Jorge.Pineda.verifier

import android.app.Activity
import android.content.Intent
import android.net.Uri
import android.support.v7.app.AppCompatActivity
import android.os.Bundle
import android.widget.Button
import android.widget.EditText
import android.widget.Toast

class Account_setting : AppCompatActivity() {

    private var link: String = ""

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.account_setting)

        val getlink = findViewById<EditText>(R.id.link)

        getlink.setText(intent.getStringExtra("link"))

        val intentR = Intent()
        val save = findViewById<Button>(R.id.save_button)
        save.setOnClickListener{
            intentR.putExtra("link",getlink.text.toString())
            setResult(Activity.RESULT_OK,intentR)
            finish()
        }
    }

/*
    fun openYoutubeLink(youtubeID: String) {
        val testIntent = Intent(Intent.ACTION_VIEW, Uri.parse(getlink.text.toString()))
        startActivity(testIntent)
    }
*/

}