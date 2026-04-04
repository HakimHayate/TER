import time
import psutil
import os
import numpy as np
import threading
from main import main

class Monitor(threading.Thread):
    def __init__(self, interval=0.5):
        super().__init__()
        self.interval = interval
        self.stop_event = threading.Event()
        self.cpu_usage = []
        self.ram_usage = []
        self.power_usage = [] # En Watts (si supporté)

    def get_power(self):
        # Lecture spécifique à Linux (Intel RAPL)
        # Note: peut nécessiter des droits sudo ou ne pas être dispo sur tous les modèles
        try:
            with open("/sys/class/power_supply/BAT0/power_now", "r") as f:
                return float(f.read()) / 1_000_000 # microwatts -> Watts
        except:
            return None

    def run(self):
        process = psutil.Process(os.getpid())
        while not self.stop_event.is_set():
            # CPU global et RAM du processus actuel + enfants
            self.cpu_usage.append(psutil.cpu_percent(interval=None))
            self.ram_usage.append(process.memory_info().rss / (1024 * 1024)) # Mo
            time.sleep(self.interval)

    def stop(self):
        self.stop_event.set()

def run_pro_benchmark(nb_warmup=3, nb_measured=7):
    all_times = []
    all_cpu = []
    all_ram = []

    total_runs = nb_warmup + nb_measured
    print(f"--- Benchmark Ressources ({nb_warmup} warmups + {nb_measured} mesures) ---")
    
    for i in range(total_runs):
        is_warmup = i < nb_warmup
        status = "WARMUP" if is_warmup else f"RUN {i - nb_warmup + 1}"
        
        monitor = Monitor()
        monitor.start()
        
        start = time.time()
        main() 
        end = time.time()
        
        monitor.stop()
        monitor.join()

        duration = end - start
        
        # On ne stocke les résultats QUE si on n'est plus en phase de chauffe
        if not is_warmup:
            all_times.append(duration)
            all_cpu.append(np.mean(monitor.cpu_usage))
            all_ram.append(np.max(monitor.ram_usage))
            
        print(f"[{status}] : {duration:.2f}s | CPU: {np.mean(monitor.cpu_usage):.1f}% | RAM: {np.max(monitor.ram_usage):.1f}Mo")

    # Affichage des statistiques finales (uniquement sur les 7 derniers)
    print("\n" + "="*45)
    print(f"RÉSULTATS FINAUX SUR {nb_measured} TESTS (après {nb_warmup} chauffes)")
    print(f"Temps moyen : {np.mean(all_times):.3f}s ± {np.std(all_times):.3f}s")
    print(f"CPU Moyen   : {np.mean(all_cpu):.1f}%")
    print(f"RAM Max Peak: {np.max(all_ram):.1f} Mo")
    print("="*45)

if __name__ == "__main__":
    # On lance 3 chauffes + 7 mesures
    run_pro_benchmark(nb_warmup=3, nb_measured=7)