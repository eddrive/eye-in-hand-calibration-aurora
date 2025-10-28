#!/usr/bin/env python3

import yaml
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.spatial.transform import Rotation as R

class InteractivePoseViewer:
    def __init__(self, pose_pairs):
        self.pose_pairs = pose_pairs
        self.current_movement_index = 0  # Indice del movimento corrente (0->1, 1->2, etc.)
        self.movement_lines = []  # Lista delle linee di movimento attuali
        
        # Configurazione plot
        self.fig = plt.figure(figsize=(12, 9))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # Disegna il plot base
        self.plot_base_poses()
        
        # Collega l'evento tastiera
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        
        # Istruzioni
        print(f"\n=== CONTROLLI INTERATTIVI ===")
        print(f"SPAZIO: Mostra movimento successivo ({len(pose_pairs)-1} movimenti totali)")
        print(f"Movimento corrente: {self.current_movement_index + 1}/{len(pose_pairs)-1}")
        print(f"Chiudi la finestra per uscire")

    def load_poses(self, yaml_file):
        """Carica le pose dal file YAML"""
        try:
            with open(yaml_file, 'r') as f:
                data = yaml.safe_load(f)
            return data['pose_pairs']
        except Exception as e:
            print(f"Errore nel caricamento del file: {e}")
            return []

    def quaternion_to_direction_vectors(self, quat):
        """Converte un quaternione [x,y,z,w] nei vettori degli assi X,Y,Z"""
        rot = R.from_quat(quat)
        rotation_matrix = rot.as_matrix()
        
        x_axis = rotation_matrix[:, 0]
        y_axis = rotation_matrix[:, 1] 
        z_axis = rotation_matrix[:, 2]
        
        return x_axis, y_axis, z_axis

    def plot_base_poses(self):
        """Disegna tutte le pose statiche (punti e assi)"""
        arrow_length = 0.02
        
        for i, pair in enumerate(self.pose_pairs):
            
            # === CAMERA (BLU) ===
            cam_pos = np.array(pair['camera']['position'])
            cam_quat = pair['camera']['orientation']
            
            self.ax.scatter(cam_pos[0], cam_pos[1], cam_pos[2], 
                          color='blue', s=50, alpha=0.7, label='Camera' if i == 0 else "")
            
            cam_x, cam_y, cam_z = self.quaternion_to_direction_vectors(cam_quat)
            
            self.ax.quiver(cam_pos[0], cam_pos[1], cam_pos[2],
                         cam_x[0] * arrow_length, cam_x[1] * arrow_length, cam_x[2] * arrow_length,
                         color='red', alpha=0.6, arrow_length_ratio=0.3)
            
            self.ax.quiver(cam_pos[0], cam_pos[1], cam_pos[2],
                         cam_y[0] * arrow_length, cam_y[1] * arrow_length, cam_y[2] * arrow_length,
                         color='green', alpha=0.6, arrow_length_ratio=0.3)
            
            self.ax.quiver(cam_pos[0], cam_pos[1], cam_pos[2],
                         cam_z[0] * arrow_length, cam_z[1] * arrow_length, cam_z[2] * arrow_length,
                         color='darkblue', alpha=0.6, arrow_length_ratio=0.3)
            
            # === SENSORE (ROSSO) ===
            sens_pos = np.array(pair['sensor']['position'])
            sens_quat = pair['sensor']['orientation']
            
            self.ax.scatter(sens_pos[0], sens_pos[1], sens_pos[2], 
                          color='red', s=50, alpha=0.7, marker='^', label='Sensore' if i == 0 else "")
            
            sens_x, sens_y, sens_z = self.quaternion_to_direction_vectors(sens_quat)
            
            self.ax.quiver(sens_pos[0], sens_pos[1], sens_pos[2],
                         sens_x[0] * arrow_length, sens_x[1] * arrow_length, sens_x[2] * arrow_length,
                         color='darkred', alpha=0.6, arrow_length_ratio=0.3)
            
            self.ax.quiver(sens_pos[0], sens_pos[1], sens_pos[2],
                         sens_y[0] * arrow_length, sens_y[1] * arrow_length, sens_y[2] * arrow_length,
                         color='darkgreen', alpha=0.6, arrow_length_ratio=0.3)
            
            self.ax.quiver(sens_pos[0], sens_pos[1], sens_pos[2],
                         sens_z[0] * arrow_length, sens_z[1] * arrow_length, sens_z[2] * arrow_length,
                         color='magenta', alpha=0.6, arrow_length_ratio=0.3)
            
            # Linea grigia tra camera e sensore della stessa acquisizione
            self.ax.plot([cam_pos[0], sens_pos[0]], 
                       [cam_pos[1], sens_pos[1]], 
                       [cam_pos[2], sens_pos[2]], 
                       'gray', alpha=0.3, linewidth=1)
            
            # Etichette numeriche
            self.ax.text(cam_pos[0], cam_pos[1], cam_pos[2] + 0.01, str(i), 
                       fontsize=8, color='blue')
            self.ax.text(sens_pos[0], sens_pos[1], sens_pos[2] + 0.01, str(i), 
                       fontsize=8, color='red')
        
        # Traiettorie complete (linee sottili)
        cam_positions = np.array([pair['camera']['position'] for pair in self.pose_pairs])
        self.ax.plot(cam_positions[:, 0], cam_positions[:, 1], cam_positions[:, 2], 
               'b-', alpha=0.3, linewidth=1, label='Traiettoria Camera')
        
        sens_positions = np.array([pair['sensor']['position'] for pair in self.pose_pairs])
        self.ax.plot(sens_positions[:, 0], sens_positions[:, 1], sens_positions[:, 2], 
               'r-', alpha=0.3, linewidth=1, label='Traiettoria Sensore')
        
        # Configurazione grafico
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        self.ax.set_zlabel('Z (m)')
        self.ax.set_title(f'Visualizzazione {len(self.pose_pairs)} Coppie di Pose\n'
                    f'Blu=Camera, Rosso=Sensore, GIALLO=Movimento Corrente')
        
        self.ax.legend()
        self.ax.grid(True, alpha=0.3)
        
        # Aspetto uguale per tutti gli assi
        max_range = np.array([cam_positions.max(), sens_positions.max()]).max()
        min_range = np.array([cam_positions.min(), sens_positions.min()]).min()
        self.ax.set_xlim([min_range, max_range])
        self.ax.set_ylim([min_range, max_range])
        self.ax.set_zlim([min_range, max_range])

    def clear_movement_lines(self):
        """Rimuove le linee di movimento precedenti"""
        for line in self.movement_lines:
            line.remove()
        self.movement_lines.clear()

    def draw_movement_lines(self, from_index, to_index):
        """Disegna le linee di movimento tra due pose consecutive"""
        if from_index >= len(self.pose_pairs) or to_index >= len(self.pose_pairs):
            return
            
        # Posizioni camera
        cam_from = np.array(self.pose_pairs[from_index]['camera']['position'])
        cam_to = np.array(self.pose_pairs[to_index]['camera']['position'])
        
        # Posizioni sensore
        sens_from = np.array(self.pose_pairs[from_index]['sensor']['position'])
        sens_to = np.array(self.pose_pairs[to_index]['sensor']['position'])
        
        # Linea movimento camera (giallo)
        cam_line = self.ax.plot([cam_from[0], cam_to[0]], 
                               [cam_from[1], cam_to[1]], 
                               [cam_from[2], cam_to[2]], 
                               'yellow', linewidth=4, alpha=0.8, label=f'Movimento Camera {from_index}→{to_index}')[0]
        
        # Linea movimento sensore (giallo)
        sens_line = self.ax.plot([sens_from[0], sens_to[0]], 
                                [sens_from[1], sens_to[1]], 
                                [sens_from[2], sens_to[2]], 
                                'gold', linewidth=4, alpha=0.8, label=f'Movimento Sensore {from_index}→{to_index}')[0]
        
        # Salva i riferimenti per poterli rimuovere dopo
        self.movement_lines.extend([cam_line, sens_line])
        
        # Calcola e stampa le distanze
        cam_distance = np.linalg.norm(cam_to - cam_from)
        sens_distance = np.linalg.norm(sens_to - sens_from)
        
        print(f"Movimento {from_index}→{to_index}:")
        print(f"  Camera dist: {cam_distance:.4f} m")
        print(f"  Sensore dist: {sens_distance:.4f} m")
        print(f"  Differenza: {abs(cam_distance - sens_distance):.4f} m")

    def on_key_press(self, event):
        """Gestisce gli eventi di tastiera"""
        if event.key == ' ':  # Barra spaziatrice
            # Rimuovi le linee precedenti
            self.clear_movement_lines()
            
            # Controlla se ci sono ancora movimenti da mostrare
            if self.current_movement_index < len(self.pose_pairs) - 1:
                # Disegna il movimento corrente
                self.draw_movement_lines(self.current_movement_index, self.current_movement_index + 1)
                
                # Avanza all'indice successivo
                self.current_movement_index += 1
                
                print(f"Movimento corrente: {self.current_movement_index}/{len(self.pose_pairs)-1}")
            else:
                # Ricomincia dal primo movimento
                self.current_movement_index = 0
                self.draw_movement_lines(0, 1)
                self.current_movement_index = 1
                print(f"Ricominciando... Movimento corrente: 1/{len(self.pose_pairs)-1}")
            
            # Ridisegna il plot
            self.fig.canvas.draw()

def print_pose_summary(pose_pairs):
    """Stampa un riassunto delle pose"""
    print(f"\n=== RIASSUNTO POSE ===")
    print(f"Numero di coppie: {len(pose_pairs)}")
    
    if pose_pairs:
        cam_positions = np.array([pair['camera']['position'] for pair in pose_pairs])
        sens_positions = np.array([pair['sensor']['position'] for pair in pose_pairs])
        
        print(f"\nCamera - Range movimento:")
        print(f"  X: {cam_positions[:, 0].min():.3f} to {cam_positions[:, 0].max():.3f} m")
        print(f"  Y: {cam_positions[:, 1].min():.3f} to {cam_positions[:, 1].max():.3f} m")
        print(f"  Z: {cam_positions[:, 2].min():.3f} to {cam_positions[:, 2].max():.3f} m")
        
        print(f"\nSensore - Range movimento:")
        print(f"  X: {sens_positions[:, 0].min():.3f} to {sens_positions[:, 0].max():.3f} m")
        print(f"  Y: {sens_positions[:, 1].min():.3f} to {sens_positions[:, 1].max():.3f} m")
        print(f"  Z: {sens_positions[:, 2].min():.3f} to {sens_positions[:, 2].max():.3f} m")

def main():
    # Nome file fisso
    yaml_file = "pose_pairs_for_rviz.yaml"
    
    # Carica pose
    try:
        with open(yaml_file, 'r') as f:
            data = yaml.safe_load(f)
        pose_pairs = data['pose_pairs']
    except Exception as e:
        print(f"Errore nel caricamento del file {yaml_file}: {e}")
        return
    
    if not pose_pairs:
        print(f"Nessuna posa trovata nel file {yaml_file}")
        return
    
    # Stampa riassunto
    print_pose_summary(pose_pairs)
    
    # Avvia visualizzazione interattiva
    print(f"\nAvvio visualizzazione interattiva di {len(pose_pairs)} coppie di pose...")
    
    viewer = InteractivePoseViewer(pose_pairs)
    plt.show()

if __name__ == '__main__':
    main()
