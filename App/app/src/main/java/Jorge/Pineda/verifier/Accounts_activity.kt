package Jorge.Pineda.verifier

import android.app.Activity
import android.content.Intent
import android.support.v7.app.AppCompatActivity
import android.os.Bundle
import android.widget.Button
import android.net.Uri
import android.widget.TextView


var link1: String = ""
var link2: String = ""
var link3: String = ""
var id1: String = ""
var id2: String = ""
var id3: String = ""

class Accounts_activity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_accounts)

        val acc1 = findViewById<Button>(R.id.add_acc_1)
        acc1.setOnClickListener{
            val myIntent1 = Intent(this,Account_setting::class.java)
            myIntent1.putExtra("id",id1)
            myIntent1.putExtra("link",link1)
            myIntent1.putExtra("account_no",1)
            startActivityForResult(myIntent1,1)
        }

        val acc2 = findViewById<Button>(R.id.add_acc_2)
        acc2.setOnClickListener{
            val myIntent2 = Intent(this,Account_setting::class.java)
            myIntent2.putExtra("id",id2)
            myIntent2.putExtra("link",link2)
            myIntent2.putExtra("account_no",2)
            startActivityForResult(myIntent2,2)
        }

        val acc3 = findViewById<Button>(R.id.add_acc_3)
        acc3.setOnClickListener{
            val myIntent3 = Intent(this,Account_setting::class.java)
            myIntent3.putExtra("id",id3)
            myIntent3.putExtra("link",link3)
            myIntent3.putExtra("account_no",3)
            startActivityForResult(myIntent3,3)
        }
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        val disp1 = findViewById<TextView>(R.id.id1)
        val disp2 = findViewById<TextView>(R.id.id2)
        val disp3 = findViewById<TextView>(R.id.id3)
        if(resultCode == Activity.RESULT_OK){
            when(requestCode){
                1 -> {
                    link1 = data!!.getStringExtra("link")
                    id1 = data!!.getStringExtra("id")
                    if(id1!=""){
                        disp1.setText(id1)
                    }
                }
                2 -> {
                    link2 = data!!.getStringExtra("link")
                    id2 = data!!.getStringExtra("id")
                    if(id2!=""){
                        disp1.setText(id2)
                    }
                }
                3 -> {
                    link3 = data!!.getStringExtra("link")
                    id3 = data!!.getStringExtra("id")
                    if(id3!=""){
                        disp1.setText(id3)
                    }
                }
            }
        }
    }
}