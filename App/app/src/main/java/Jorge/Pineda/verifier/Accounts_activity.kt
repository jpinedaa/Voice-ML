package Jorge.Pineda.verifier

import android.app.Activity
import android.content.Intent
import android.support.v7.app.AppCompatActivity
import android.os.Bundle
import android.widget.Button
import android.net.Uri
import android.util.Log
import android.view.View
import android.widget.TextView
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.lang.Exception


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

        val ids_data = File(filesDir, "ids_data")
        if(!ids_data.isFile){
            Log.d("DEBUG", "Location 1")
            var ids_json = JSONObject()
            var users = JSONArray()
            for (i in 0 until 3) {
                var user = JSONObject()
                user.put("id", 0)
                user.put("link", 0)
                users.put(user)
            }
            ids_json.put("users",users)
            ids_data.writeText(ids_json.toString())

        }

        Log.d("DEBUG", "Location 2")
        val disp1 = findViewById<TextView>(R.id.id1)
        val disp2 = findViewById<TextView>(R.id.id2)
        val disp3 = findViewById<TextView>(R.id.id3)
        val disp = arrayOf(disp1,disp2,disp3)


        val ids_obj = JSONObject(ids_data.readText())
        var ids_array  = ids_obj.get("users")
        ids_array = ids_array as JSONArray
        for (i in 0 until 3){
            var user = ids_array[i]
            user = user as JSONObject
            var id = user.get("id")
            if (id != 0){
                id = id as String
                var link = user.get("link")
                link = link as String
                disp[i].setText(id)
                when(i){
                    0->{
                        id1 = id
                        link1 = link
                    }
                    1->{
                        id2 = id
                        link2 = link
                    }
                    2->{
                        id3 = id
                        link3 = link
                    }
                }
                Log.d("ids", i.toString() + " value: " + id + " next " + link)
            }

        }


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

    public override fun onResume() {
        super.onResume()
        // put your code here...
        val disp1 = findViewById<TextView>(R.id.id1)
        val disp2 = findViewById<TextView>(R.id.id2)
        val disp3 = findViewById<TextView>(R.id.id3)
        val disp = arrayOf(disp1,disp2,disp3)
        val ids = arrayOf(id1,id2,id3)
        for (i in 0 until 3){
            if (ids[i] != ""){
                disp[i].setText(ids[i])
            }else {
                disp[i].setText(R.string.no_account)
            }
        }

    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        val disp1 = findViewById<TextView>(R.id.id1)
        val disp2 = findViewById<TextView>(R.id.id2)
        val disp3 = findViewById<TextView>(R.id.id3)
        if(resultCode == Activity.RESULT_OK){
            val ids_data = File(filesDir, "ids_data")
            val ids_obj = JSONObject(ids_data.readText())
            var ids_array  = ids_obj.get("users")
            ids_array = ids_array as JSONArray
            when(requestCode){
                1 -> {
                    try {
                        link1 = data!!.getStringExtra("link")
                        id1 = data!!.getStringExtra("id")
                    }catch (e: Exception){
                        link1 = ""
                        id1 = ""
                    }
                    if(id1!=""){
                        disp1.setText(id1)
                        val user = JSONObject()
                        user.put("id", id1)
                        user.put("link", link1)
                        ids_array.put(0, user)
                        val new_json = JSONObject()
                        new_json.put("users", ids_array)
                        ids_data.delete()
                        ids_data.writeText(new_json.toString())
                    }
                }
                2 -> {
                    try {
                        link2 = data!!.getStringExtra("link")
                        id2 = data!!.getStringExtra("id")
                    }catch (e: Exception){
                        link2 = ""
                        id2 = ""
                    }
                    if(id2!=""){
                        disp1.setText(id2)
                        val user = JSONObject()
                        user.put("id", id2)
                        user.put("link", link2)
                        ids_array.put(1, user)
                        val new_json = JSONObject()
                        new_json.put("users", ids_array)
                        ids_data.delete()
                        ids_data.writeText(new_json.toString())
                    }
                }
                3 -> {
                    try {
                        link3 = data!!.getStringExtra("link")
                        id3 = data!!.getStringExtra("id")
                    } catch (e : Exception){
                        link3 = ""
                        id3 = ""
                    }
                    if(id3!=""){
                        disp1.setText(id3)
                        val user = JSONObject()
                        user.put("id", id3)
                        user.put("link", link3)
                        ids_array.put(2, user)
                        val new_json = JSONObject()
                        new_json.put("users", ids_array)
                        ids_data.delete()
                        ids_data.writeText(new_json.toString())
                    }
                }
            }
        }
    }
}