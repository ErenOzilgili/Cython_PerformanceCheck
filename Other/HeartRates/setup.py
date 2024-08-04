from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

# definitions


extensions = [Extension('wrap', sources=['wrapper.pyx', 'HeartRates.cpp'], language='c++')]
#'wrap' --> nasıl importta atfedileceği (test dosyasına bak) --> from wrap import *
"""
Derlenecek c/cython modüllerini belirtmek için,
ext_modules = cythonize(extensions) da kullanılır
"""

setup(
    name = 'my_package', #Başkaları tarafından kullanım, tanımlamak ve dağıtmak için gerekli
    version = '1.0', #Sürüm kontrolünü sağlamak için, gerekli durumda paket güncellemeleri için
    description = 'Dot product almak için küçük bir kod', #Küçük açıklama için kullanılır
    author='Eren Özilgili',
    author_email='erenozilgili@gmail.com', #İletişim bilgileri
    url='https://github.com/johndoe/my_package', #Kaynak kodu için bağlantı
    
        #Package, find_package, find_package(where = ''),
        #find_package(include, exclude) ---> Gözden geçir
    
    ext_modules=cythonize(extensions) #Derlenecek c/cython modülleri içindir
)