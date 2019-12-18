package io.lies.pytorchdebug

import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.util.Log
import org.pytorch.IValue
import org.pytorch.Module
import org.pytorch.Tensor
import java.io.File

class MainActivity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        Log.d("PytorchDebug", "Loading PyTorch components")

        val inputHeight = 299
        val inputWidth = 299
        val inputTensorBuffer = Tensor.allocateFloatBuffer(3 * inputWidth * inputHeight)
        val inputTensor = Tensor.fromBlob( inputTensorBuffer, longArrayOf( 1, 3, inputHeight.toLong(), inputWidth.toLong()))
        val modulePath = File(baseContext.filesDir, "model.pt").absolutePath
        var module : Module? = null

        if (!File(baseContext.filesDir, "model.pt").exists() ) {
            getModuleFile(modulePath)
            Log.d("PytorchDebug", "Module loaded")
        }
        try {
            module = Module.load(modulePath)
        } catch (e: Exception) {
            e.printStackTrace()
        }

        val outputTensor = module?.forward(IValue.from(inputTensor))?.toTensor()

        Log.d("PytorchDebug", outputTensor.toString())
    }


    private fun getModuleFile(modulePath: String) {
        val modelInput  =
            baseContext.assets.open("model.pt")
        val modelOutput = File(modulePath).outputStream()
        modelInput.copyTo(modelOutput)
        modelInput.close()
        modelOutput.close()
    }
}
