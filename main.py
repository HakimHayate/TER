import configuration as cf
from utils import  read_json , stream_pdf_pages
from preprocess.preprocess import Preprocess
from identification.identification import Identification
import multiprocessing as mp
import fitz

"""
Module principal de traitement haute performance pour Hekzam-FormDetection.

Ce script implémente une architecture Producteur-Consommateur (Producer-Consumer) 
utilisant la bibliothèque multiprocessing de Python. Il permet de paralléliser 
le rendu des pages PDF et le prétraitement des images sur plusieurs cœurs CPU, 
tout en alimentant un moteur d'identification ONNX.
"""

def consumer(batch_queue):
    """
    Processus Consommateur : Récupère les lots d'images et lance l'identification.
    
    Il écoute la file d'attente (Queue) et transmet les lots au moteur ONNX 
    jusqu'à la réception du signal d'arrêt (None).
    """
    identification = Identification(cf.model_path)
    identification.fast_identify(iter(batch_queue.get, None))  
    identification.draw_confusion_matrix()

    
def producer(pdf_path, rects, start_page, end_page, batch_queue):
    """
    Processus Producteur : Extrait, recalage et découpage des pages du PDF.
    
    Chaque producteur gère une portion spécifique du document. Il transforme 
    les pages en images, applique l'homographie et le formatage MNIST, puis 
    dépose les résultats dans la file d'attente partagée.
    """
    img_generator = stream_pdf_pages(pdf_path, start_page=start_page, end_page=end_page)
    src, _ = read_json(cf.json_path) 
    preprocess = Preprocess(src)
        
    for batch in preprocess.get_streaming_batches(img_generator, rects, save_dir=cf.identify_path, start_page_idx=start_page):
        batch_queue.put(batch)


def main() :
    """
    Orchestrateur du système multiprocessus.
    
    - Divise le PDF en 3 segments équitables (0 -> 1/3 -> 2/3 -> Fin).
    - Lance 3 producteurs en parallèle pour saturer le CPU en prétraitement.
    - Initialise un consommateur pour l'inférence centralisée.
    - Gère la synchronisation (Join) et la fermeture propre de la file d'attente.
    """
    batch_queue = mp.Queue(maxsize=10)  

    with fitz.open(cf.pdf_path) as doc:
        num_pages = len(doc)
    mid1 = num_pages // 3
    mid2 = 2 * num_pages // 3
    _ ,rects = read_json(cf.json_path)

    p1 = mp.Process(target=producer, args=(cf.pdf_path, rects, 0, mid1, batch_queue))
    p2 = mp.Process(target=producer, args=(cf.pdf_path, rects, mid1, mid2, batch_queue))
    p3 = mp.Process(target=producer, args=(cf.pdf_path, rects, mid2, num_pages, batch_queue))
    p1.start()
    p2.start()
    p3.start()
    num_consumers = 1
    process = []
    for _ in range(num_consumers):
        p = mp.Process(target=consumer, args=(batch_queue,))
        p.start()
        process.append(p)

    p1.join()
    p2.join()
    p3.join()

    for _ in range(num_consumers):
        batch_queue.put(None)

    for p in process:
        p.join()


if __name__ == "__main__" :
    main()
