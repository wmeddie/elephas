
import os

def to_array(collection, t):
    from pyspark.context import SparkContext

    sc = SparkContext._active_spark_context
    jgateway = sc._gateway
    jtype = None

    if t is int:
        jtype = jgateway.jvm.int
    elif t is long:
        jtype = jgateway.jvm.long

    if jtype is not None:
        a = jgateway.new_array(jtype, len(collection))
        i = 0
        for item in collection:
            a[i] = item
            i += 1

        return a
    else:
        return ArrayList(collection).toArray()

def _py4jclass(cls):
    from pyspark.context import SparkContext

    sc = SparkContext._active_spark_context
    names = cls.jvm_cls_name.split('.')
    last = sc._jvm
    for name in names:
        last = getattr(last, name)
    return last

def _jniusclass(cls):
    import pydl4j

    pydl4j.validate_jars()
    pydl4j.add_classpath(os.getcwd())

    from jnius import autoclass
    return autoclass(cls.jvm_cls_name)

class JvmMetaClass(type):
    def __call__(cls, *args, **kwargs):
        try:
            return _py4jclass(cls)(*args, **kwargs)
        except:
            return _jniusclass(cls)(*args, **kwargs)

    def __getattr__(cls, key):
        try:
            return getattr(_py4jclass(cls), key)
        except:
            try:
                return getattr(_jniusclass(cls), key)
            except:
                raise Exception("Unable to get jvm class: %s" % cls.jvm_cls_name)

# Java
class File(object):
    __metaclass__ = JvmMetaClass
    jvm_cls_name = 'java.io.File'

class ClassLoader(object):
    __metaclass__ = JvmMetaClass
    jvm_cls_name = 'java.lang.ClassLoader'

class ArrayList(object):
    __metaclass__ = JvmMetaClass
    jvm_cls_name = 'java.util.ArrayList'

class Arrays(object):
    __metaclass__ = JvmMetaClass
    jvm_cls_name = 'java.util.Arrays'

class String(object):
    __metaclass__ = JvmMetaClass
    jvm_cls_name = 'java.lang.String'

class System(object):
    __metaclass__ = JvmMetaClass
    jvm_cls_name = 'java.lang.System'

class Integer(object):
    __metaclass__ = JvmMetaClass
    jvm_cls_name = 'java.lang.Integer'

class Float(object):
    __metaclass__ = JvmMetaClass
    jvm_cls_name = 'java.lang.Float'

class Double(object):
    __metaclass__ = JvmMetaClass
    jvm_cls_name = 'java.lang.Double'

# JavaCPP
class DoublePointer(object):
    __metaclass__ = JvmMetaClass
    jvm_cls_name = 'org.bytedeco.javacpp.DoublePointer'

class FloatPointer(object):
    __metaclass__ = JvmMetaClass
    jvm_cls_name = 'org.bytedeco.javacpp.FloatPointer'

class IntPointer(object):
    __metaclass__ = JvmMetaClass
    jvm_cls_name = 'org.bytedeco.javacpp.IntPointer'


# Spark
class SparkContext(object):
    __metaclass__ = JvmMetaClass
    jvm_cls_name = 'org.apache.spark.SparkContext'

class JavaSparkContext(object):
    __metaclass__ = JvmMetaClass
    jvm_cls_name = 'org.apache.spark.api.java.JavaSparkContext'

class SparkConf(object):
    __metaclass__ = JvmMetaClass
    jvm_cls_name = 'org.apache.spark.SparkConf'


# ND4J
class Nd4j(object):
    __metaclass__ = JvmMetaClass
    jvm_cls_name = 'org.nd4j.linalg.factory.Nd4j'

class INDArray(object):
    __metaclass__ = JvmMetaClass
    jvm_cls_name = 'org.nd4j.linalg.api.ndarray.INDArray'

class Transforms(object):
    __metaclass__ = JvmMetaClass
    jvm_cls_name = 'org.nd4j.linalg.ops.transforms.Transforms'

class NDArrayIndex(object):
    __metaclass__ = JvmMetaClass
    jvm_cls_name = 'org.nd4j.linalg.indexing.NDArrayIndex'

class DataBuffer(object):
    __metaclass__ = JvmMetaClass
    jvm_cls_name = 'org.nd4j.linalg.api.buffer.DataBuffer'

class Shape(object):
    __metaclass__ = JvmMetaClass
    jvm_cls_name = 'org.nd4j.linalg.api.shape.Shape'

class BinarySerde(object):
    __metaclass__ = JvmMetaClass
    jvm_cls_name = 'org.nd4j.serde.binary.BinarySerde'

class DataTypeUtil(object):
    __metaclass__ = JvmMetaClass
    jvm_cls_name = 'org.nd4j.linalg.api.buffer.util.DataTypeUtil'

class DataType(object):
    __metaclass__ = JvmMetaClass
    jvm_cls_name = 'org.nd4j.linalg.api.buffer.DataType'

class NativeOpsHolder(object):
    __metaclass__ = JvmMetaClass
    jvm_cls_name = 'org.nd4j.nativeblas.NativeOpsHolder'

class DataSet(object):
    __metaclass__ = JvmMetaClass
    jvm_cls_name = 'org.nd4j.linalg.dataset.DataSet'

# Import
class KerasModelImport(object):
    __metaclass__ = JvmMetaClass
    jvm_cls_name  = 'org.deeplearning4j.nn.modelimport.keras.KerasModelImport'

class ElephasModelImport(object):
    __metaclass__ = JvmMetaClass
    jvm_cls_name = 'org.deeplearning4j.spark.parameterserver.modelimport.elephas.ElephasModelImport'
