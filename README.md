# Face detector

Con el objetivo de obtener más información sobre el estado del conductor durante
la simulación, se desarrolló un componente encargado de monitorear la vista del
usuario a través de una cámara. Esto es muy valioso ya que complementa el objetivo
de las tareas DRT y ayuda a identificar la atención del conductor durante las pruebas.

Para el desarrollo de este componente se utilizó la librería
[OpenCV](https://opencv.org/), la cual brinda una gran cantidad de funcionalidades
para la detección de elementos y personas en tiempo real. De esta forma, utilizando
el mecanísmo de "Haar Cascade Classifier" es posible identificar objetos en imagenes
o videos, sin importar su ubicación ni tamaño en la imagen. Además este algoritmo
es muy rápido, lo cual nos permite detectar elementos en tiempo real.
