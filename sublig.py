import numpy as np
import sys
import os
import argparse


def parser():
    parser = argparse.ArgumentParser(description='Substitute ligands in a molecule')
    parser.add_argument('INPUT', help='Input file  (.xyz file)', type=str)
    parser.add_argument('-id', '--input_donor_atom', nargs="*", type=str, help='donor atom label number', required=True)
    parser.add_argument('-m', '--input_metal_atom', type=str, help='metal atom label number', required=True)# acceptor atom
    parser.add_argument('-l', '--ligand', type=str, help='Ligand to substitute (.xyz file)', required=True)
    parser.add_argument('-sd', '--sub_donor_atom', nargs="*", type=str, help='sub atom label number', required=True)
    
    args = parser.parse_args()
    args = vars(args)
    return args

def num_parse(numbers):
    sub_list = []

    sub_tmp_list = numbers.split(",")
    for sub in sub_tmp_list:                        
        if "-" in sub:
            for j in range(int(sub.split("-")[0]),int(sub.split("-")[1])+1):
                sub_list.append(j)
        else:
            sub_list.append(int(sub))    
    return sub_list




def covalent_radii_lib(element):
    if element is int:
        element = number_element(element)
    CRL = {"H": 0.32, "He": 0.46, 
           "Li": 1.33, "Be": 1.02, "B": 0.85, "C": 0.75, "N": 0.71, "O": 0.63, "F": 0.64, "Ne": 0.67, 
           "Na": 1.55, "Mg": 1.39, "Al":1.26, "Si": 1.16, "P": 1.11, "S": 1.03, "Cl": 0.99, "Ar": 0.96, 
           "K": 1.96, "Ca": 1.71, "Sc": 1.48, "Ti": 1.36, "V": 1.34, "Cr": 1.22, "Mn": 1.19, "Fe": 1.16, "Co": 1.11, "Ni": 1.10, "Cu": 1.12, "Zn": 1.18, "Ga": 1.24, "Ge": 1.24, "As": 1.21, "Se": 1.16, "Br": 1.14, "Kr": 1.17, 
           "Rb": 2.10, "Sr": 1.85, "Y": 1.63, "Zr": 1.54,"Nb": 1.47,"Mo": 1.38,"Tc": 1.28,"Ru": 1.25,"Rh": 1.25,"Pd": 1.20,"Ag": 1.28,"Cd": 1.36,"In": 1.42,"Sn": 1.40,"Sb": 1.40,"Te": 1.36,"I": 1.33,"Xe": 1.31,
           "Cs": 2.32,"Ba": 1.96,"La":1.80,"Ce": 1.63,"Pr": 1.76,"Nd": 1.74,"Pm": 1.73,"Sm": 1.72,"Eu": 1.68,"Gd": 1.69 ,"Tb": 1.68,"Dy": 1.67,"Ho": 1.66,"Er": 1.65,"Tm": 1.64,"Yb": 1.70,"Lu": 1.62,"Hf": 1.52,"Ta": 1.46,"W": 1.37,"Re": 1.31,"Os": 1.29,"Ir": 1.22,"Pt": 1.23,"Au": 1.24,"Hg": 1.33,"Tl": 1.44,"Pb":1.44,"Bi":1.51,"Po":1.45,"At":1.47,"Rn":1.42, 'X':1.000}#ang.
    # ref. Pekka Pyykkö; Michiko Atsumi (2009). “Molecular single-bond covalent radii for elements 1 - 118”. Chemistry: A European Journal 15: 186–197. doi:10.1002/chem.200800987. (H...Rn)
            
    return CRL[element] / UnitValueLib().bohr2angstroms#Bohr


def number_element(num):
    elem = {1: "H",  2:"He",
         3:"Li", 4:"Be", 5:"B", 6:"C", 7:"N", 8:"O", 9:"F", 10:"Ne", 
        11:"Na", 12:"Mg", 13:"Al", 14:"Si", 15:"P", 16:"S", 17:"Cl", 18:"Ar",
        19:"K", 20:"Ca", 21:"Sc", 22:"Ti", 23:"V", 24:"Cr", 25:"Mn", 26:"Fe", 27:"Co", 28:"Ni", 29:"Cu", 30:"Zn", 31:"Ga", 32:"Ge", 33:"As", 34:"Se", 35:"Br", 36:"Kr",
        37:"Rb", 38:"Sr", 39:"Y", 40:"Zr", 41:"Nb", 42:"Mo",43:"Tc",44:"Ru",45:"Rh", 46:"Pd", 47:"Ag", 48:"Cd", 49:"In", 50:"Sn", 51:"Sb", 52:"Te", 53:"I", 54:"Xe",
        55:"Cs", 56:"Ba", 57:"La",58:"Ce",59:"Pr",60:"Nd",61:"Pm",62:"Sm", 63:"Eu", 64:"Gd", 65:"Tb", 66:"Dy" ,67:"Ho", 68:"Er", 69:"Tm", 70:"Yb", 71:"Lu", 72:"Hf", 73:"Ta", 74:"W", 75:"Re", 76:"Os", 77:"Ir", 78:"Pt", 79:"Au", 80:"Hg", 81:"Tl", 82:"Pb", 83:"Bi", 84:"Po", 85:"At", 86:"Rn"}
        
    return elem[num]


def check_atom_connectivity(mol_list, element_list, atom_num, covalent_radii_threshold_scale=1.2):
    connected_atoms = [atom_num]
    searched_atoms = []
    while True:
        for i in connected_atoms:
            if i in searched_atoms:
                continue
            
            for j in range(len(mol_list)):
                dist = np.linalg.norm(np.array(mol_list[i], dtype="float64") - np.array(mol_list[j], dtype="float64"))
                
                covalent_dist_threshold = covalent_radii_threshold_scale * (covalent_radii_lib(element_list[i]) + covalent_radii_lib(element_list[j]))
                
                if dist < covalent_dist_threshold:
                    if not j in connected_atoms:
                        connected_atoms.append(j)
            
            searched_atoms.append(i)
     
        if len(connected_atoms) == len(searched_atoms):
            break
    
    return sorted(connected_atoms)


def make_lig_fragment_list(mol_list, element_list, metal, lig_donor, covalent_radii_threshold_scale=1.2):
    
    
    tmp_mol_list = mol_list / UnitValueLib().bohr2angstroms
    for i in range(len(metal)):
        metal[i] -= 1
        tmp_mol_list[metal[i]] += np.array([np.inf, np.inf, np.inf], dtype="float64") 
    atom_label_list = [i for i in range(len(mol_list)) if not i in metal]
    for i in range(len(lig_donor)):
        for j in range(len(lig_donor[i])):
            lig_donor[i][j] -= 1


    lig_fragment_list = []
    while len(atom_label_list) > 0:
        tmp_fragm_list = check_atom_connectivity(tmp_mol_list, element_list, atom_label_list[0], covalent_radii_threshold_scale)
        for j in tmp_fragm_list:
            if not j in metal:
                atom_label_list.remove(j)
         
        lig_fragment_list.append(tmp_fragm_list)
    
    substitution_tgt_lig = []
    for lig in lig_fragment_list:
        for donor in lig_donor:
            for d in donor:
                if d in lig:
                    substitution_tgt_lig.extend(lig)    

    #print("\ncoordinated ligands list:", lig_fragment_list)
    #print("tgt ligand:", substitution_tgt_lig)
    return lig_fragment_list, substitution_tgt_lig



def read_xyz(file_path):
    with open(file_path, "r") as f:
        words = f.read().splitlines()

    coord = []
    element_list = []
    for i in range(2, len(words)):
        if len(words[i].split()) == 4:
            element_list.append(words[i].split()[0])
            coord.append(words[i].split()[1:4])
    coord = np.array(coord, dtype="float64")
    return coord, element_list


def save_xyz(file_path, coord, element_list, add_name):
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    natom = len(coord)
    save_file_name = base_name + "_" + add_name + ".xyz"
    with open(save_file_name, "w") as f:
        f.write(str(natom)+"\n")
        f.write(add_name+"\n")
        for i in range(len(coord)):
            f.write(element_list[i]+"  {0:.12f}   {1:.12f}   {2:.12f}".format(*coord[i].tolist())+"\n")


    return

class UnitValueLib: 
    def __init__(self):
        self.bohr2angstroms = 0.52917721067
        return
        


class SubstituteLigand():
    def __init__(self, **kwargs):
        self.complex_abs_path = os.path.abspath(kwargs["INPUT"]) # xyz file
        input_donor_atom = kwargs["input_donor_atom"]
        input_metal_atom = kwargs["input_metal_atom"]
        self.sub_ligand_abs_path = os.path.abspath(kwargs["ligand"]) # xyz file
        sub_donor_atom = kwargs["sub_donor_atom"]

        self.donor_atom = []
        self.sub_donor_atom = []

        for i in range(len(input_donor_atom)):
            self.donor_atom.append(num_parse(input_donor_atom[i])) # for hapticity > 1
            self.sub_donor_atom.append(num_parse(sub_donor_atom[i])) # for hapticity > 1
        
        


        self.metal_atom = num_parse(input_metal_atom) # for polynuclear complex
        self.ndonor = len(self.donor_atom)
        self.nsub_donor = len(self.sub_donor_atom)
        assert self.ndonor == self.nsub_donor, "The number of donor atoms and the number of substituting atoms must be the same."

        self.covalent_radii_threshold_scale = 1.2
        self.iteration = 10000
        return


    def make_rotmat_vec2z(self, vec):
        unit_vec = vec / np.linalg.norm(vec)
        z_vec = np.array([0.0, 0.0, 1.0], dtype="float64")
        u = np.cross(unit_vec, z_vec)
        cos_theta = np.dot(unit_vec, z_vec)
        sin_theta = np.linalg.norm(u)
        if sin_theta == 0:
            return np.eye(3)
        u_normalized = u / np.linalg.norm(u)
        rotation_matrix = (
            np.eye(3) * cos_theta +
            np.outer(u_normalized, u_normalized) * (1 - cos_theta) +
            np.array([
                [0, -u_normalized[2], u_normalized[1]],
                [u_normalized[2], 0, -u_normalized[0]],
                [-u_normalized[1], u_normalized[0], 0]
            ]) * sin_theta
        )
        return rotation_matrix

    def generate_rotmat(self, x_angle, y_angle, z_angle):
        cos_x_angle = np.cos(x_angle)
        sin_x_angle = np.sin(x_angle)
        x_rotmat = np.array([[1, 0, 0],
                             [0, cos_x_angle, -sin_x_angle],
                             [0, sin_x_angle, cos_x_angle]],
                             dtype="float64")
        
        cos_y_angle = np.cos(y_angle)
        sin_y_angle = np.sin(y_angle)

        y_rotmat = np.array([[cos_y_angle, 0, sin_y_angle],
                             [0, 1, 0],
                             [-sin_y_angle, 0, cos_y_angle]],
                               dtype="float64")

        cos_z_angle = np.cos(z_angle)
        sin_z_angle = np.sin(z_angle)

        z_rotmat = np.array([[cos_z_angle, -sin_z_angle, 0],
                             [sin_z_angle, cos_z_angle, 0],
                             [0, 0, 1]]
                            ,dtype="float64")

        rotmat = np.dot(np.dot(z_rotmat, y_rotmat), x_rotmat)

        return rotmat



    def calc_LJ_pot(self, donor_centerized_sub_ligand_coord, tgt_removed_z_vec_complex_coord):


        dist_mat = np.linalg.norm(donor_centerized_sub_ligand_coord[:, np.newaxis] - tgt_removed_z_vec_complex_coord, axis=2)

        LJ_potential = np.sum(1.0 * ((1.0 / dist_mat) ** 12 -2 * (1.0 / dist_mat) ** 6) + (1e-5 / dist_mat))

        return LJ_potential

    def l_bfgs(self, prev_inv_hess, grad, prev_grad, prev_step, delta = 1.0):
        
        delta_grad = grad - prev_grad
        rho_inv = np.sum(delta_grad * prev_step)  
        if rho_inv > 1e-10:
            rho = 1.0 / rho_inv
            A = (np.eye(len(grad)) - rho * np.outer(prev_step, delta_grad))
            inv_hess = np.dot(np.dot(A, prev_inv_hess), A) + rho * np.outer(prev_step, prev_step)
            step = np.dot(inv_hess, grad.T).T
        else:
            step = delta * grad
            inv_hess = prev_inv_hess

        
        return step, inv_hess



    def opt_place(self, tgt_removed_z_vec_complex_coord, init_distance, donor_centerized_sub_ligand_coord):
        delta = 0.0001
        lr = 0.01
        z_vec = np.array([0, 0, init_distance], dtype="float64")
        inv_hess = np.eye(3)

        for i in range(self.iteration):


            tmp_sub_lig_coord = z_vec + donor_centerized_sub_ligand_coord
            lj_pot = self.calc_LJ_pot(tmp_sub_lig_coord, tgt_removed_z_vec_complex_coord)
            if i % 50 == 0:
                print("Iteration "+str(i)+" : ", lj_pot)

            m_x_rotmat = self.generate_rotmat(-delta, 0, 0)
            p_x_rotmat = self.generate_rotmat(delta, 0, 0)
            m_y_rotmat = self.generate_rotmat(0, -delta, 0)
            p_y_rotmat = self.generate_rotmat(0, delta, 0)
            m_z_rotmat = self.generate_rotmat(0, 0, -delta)
            p_z_rotmat = self.generate_rotmat(0, 0, delta)
        
            xm_donor_centerized_sub_ligand_coord = np.dot(m_x_rotmat, donor_centerized_sub_ligand_coord.T).T
            
            tmp_sub_lig_coord = z_vec + xm_donor_centerized_sub_ligand_coord
            xm_lj_pot = self.calc_LJ_pot(tmp_sub_lig_coord, tgt_removed_z_vec_complex_coord)


            xp_donor_centerized_sub_ligand_coord = np.dot(p_x_rotmat, donor_centerized_sub_ligand_coord.T).T
            tmp_sub_lig_coord = z_vec + xp_donor_centerized_sub_ligand_coord
            xp_lj_pot = self.calc_LJ_pot(tmp_sub_lig_coord, tgt_removed_z_vec_complex_coord)

            ym_donor_centerized_sub_ligand_coord = np.dot(m_y_rotmat, donor_centerized_sub_ligand_coord.T).T
            tmp_sub_lig_coord = z_vec + ym_donor_centerized_sub_ligand_coord
            ym_lj_pot = self.calc_LJ_pot(tmp_sub_lig_coord, tgt_removed_z_vec_complex_coord)


            yp_donor_centerized_sub_ligand_coord = np.dot(p_y_rotmat, donor_centerized_sub_ligand_coord.T).T
            tmp_sub_lig_coord = z_vec + yp_donor_centerized_sub_ligand_coord
            yp_lj_pot = self.calc_LJ_pot(tmp_sub_lig_coord, tgt_removed_z_vec_complex_coord)

            zm_donor_centerized_sub_ligand_coord = np.dot(m_z_rotmat, donor_centerized_sub_ligand_coord.T).T
            tmp_sub_lig_coord = z_vec + zm_donor_centerized_sub_ligand_coord
            zm_lj_pot = self.calc_LJ_pot(tmp_sub_lig_coord, tgt_removed_z_vec_complex_coord)


            zp_donor_centerized_sub_ligand_coord = np.dot(p_z_rotmat, donor_centerized_sub_ligand_coord.T).T
            tmp_sub_lig_coord = z_vec + zp_donor_centerized_sub_ligand_coord
            zp_lj_pot = self.calc_LJ_pot(tmp_sub_lig_coord, tgt_removed_z_vec_complex_coord)

            x_angle_grad = (xp_lj_pot - xm_lj_pot) / (2 * delta)
            y_angle_grad = (yp_lj_pot - ym_lj_pot) / (2 * delta)
            z_angle_grad = (zp_lj_pot - zm_lj_pot) / (2 * delta)
            if i % 50 == 0:
                print("grad : ", x_angle_grad, y_angle_grad, z_angle_grad)

            if i == 0:
                grad_rotmat = self.generate_rotmat(lr * x_angle_grad, lr * y_angle_grad, lr * z_angle_grad)
                prev_step = np.array([lr * x_angle_grad, lr * y_angle_grad, lr * z_angle_grad], dtype="float64")
                prev_grad = np.array([x_angle_grad, y_angle_grad, z_angle_grad], dtype="float64")
            else:
                grad = np.array([x_angle_grad, y_angle_grad, z_angle_grad], dtype="float64")
                step, inv_hess = self.l_bfgs(inv_hess, grad, prev_grad, prev_step)
                norm_step = np.linalg.norm(step)
                step = min(norm_step, 1.0) * step / norm_step
                grad_rotmat = self.generate_rotmat(step[0].item(), step[1].item(), step[2].item())
                prev_step = np.array([[step[0].item(), step[1].item(), step[2].item()]], dtype="float64")
                prev_grad = np.array([x_angle_grad, y_angle_grad, z_angle_grad], dtype="float64")

            
            grad_rotmat = self.generate_rotmat(lr * x_angle_grad, lr * y_angle_grad, lr * z_angle_grad)

            donor_centerized_sub_ligand_coord = np.dot(grad_rotmat, donor_centerized_sub_ligand_coord.T).T



            if np.linalg.norm(np.array([x_angle_grad, y_angle_grad, z_angle_grad], dtype="float64")) < 1e-10:
                print("grad : ", np.linalg.norm(np.array([x_angle_grad, y_angle_grad, z_angle_grad], dtype="float64")))
                print("Converged at itr.", i)
                break

        z_vec_donor_centerized_sub_ligand_coord = z_vec + donor_centerized_sub_ligand_coord
        print("---------------")
        return z_vec_donor_centerized_sub_ligand_coord

    def opt_place_bidentate(self, tgt_removed_z_vec_complex_coord, donor_centerized_sub_ligand_coord):
        delta = 0.0001
        lr = 0.1
        inv_hess = np.eye(1)

        for i in range(self.iteration):

            lj_pot = self.calc_LJ_pot(donor_centerized_sub_ligand_coord, tgt_removed_z_vec_complex_coord)
            if i % 50 == 0:
                print("Iteration "+str(i)+" : ", lj_pot)

            m_z_rotmat = self.generate_rotmat(0, 0, -delta)
            p_z_rotmat = self.generate_rotmat(0, 0, delta)
            zm_donor_centerized_sub_ligand_coord = np.dot(m_z_rotmat, donor_centerized_sub_ligand_coord.T).T
           
            zm_lj_pot = self.calc_LJ_pot(zm_donor_centerized_sub_ligand_coord, tgt_removed_z_vec_complex_coord)


            zp_donor_centerized_sub_ligand_coord = np.dot(p_z_rotmat, donor_centerized_sub_ligand_coord.T).T
            
            zp_lj_pot = self.calc_LJ_pot(zp_donor_centerized_sub_ligand_coord, tgt_removed_z_vec_complex_coord)

            z_angle_grad =  (zp_lj_pot - zm_lj_pot) / (2 * delta)
           
            if i % 50 == 0:
                print("grad : ", z_angle_grad)
            

            if i == 0:
                grad_rotmat = self.generate_rotmat(0, 0, lr * z_angle_grad)
                prev_step = np.array([lr * z_angle_grad], dtype="float64")
                prev_grad = np.array([z_angle_grad], dtype="float64")
            else:
                grad = np.array([z_angle_grad], dtype="float64")
                step, inv_hess = self.l_bfgs(inv_hess, grad, prev_grad, prev_step)
                norm_step = np.linalg.norm(step)
                step = min(norm_step, 1.0) * step / norm_step
                grad_rotmat = self.generate_rotmat(0, 0, step.item())
                prev_step = np.array([[step.item()]], dtype="float64")
                prev_grad = np.array([z_angle_grad], dtype="float64")



            donor_centerized_sub_ligand_coord = np.dot(grad_rotmat, donor_centerized_sub_ligand_coord.T).T

            if np.linalg.norm(np.array([z_angle_grad], dtype="float64")) < 1e-10:
                print("grad : ", z_angle_grad)
                print("Converged at itr.", i)
                break

        print("---------------")
        return donor_centerized_sub_ligand_coord






    def calc_center(self, coord):
        center = np.array([0.0, 0.0, 0.0], dtype="float64")

        for i in range(len(coord)):
            center += coord[i]
        center /= len(coord)
        return center

    def run_monodentate_lig(self):
        metal_centerized_complex_coord = self.complex_coord - self.metal_center_coord
        metal2input_donor_vec = self.input_donor_center_coord - self.metal_center_coord
        z_rot_mat = self.make_rotmat_vec2z(metal2input_donor_vec)

        z_vec_metal_centerized_complex_coord = np.dot(z_rot_mat, metal_centerized_complex_coord.T).T



        
        donor_centerized_sub_ligand_coord = self.sub_ligand_coord - self.sub_donor_center_coord

        #------
        mean_metal_covalent_radii = 0.0
        for i in self.metal_atom:
            mean_metal_covalent_radii += covalent_radii_lib(self.complex_element_list[i - 1])
        mean_metal_covalent_radii /= len(self.metal_atom)

        mean_sub_donor_covalent_radii = 0.0
        for i in self.sub_donor_atom[0]:
            mean_sub_donor_covalent_radii += covalent_radii_lib(self.sub_ligand_element_list[i - 1])
        mean_sub_donor_covalent_radii /= len(self.sub_donor_atom[0])

        init_distance = mean_metal_covalent_radii + mean_sub_donor_covalent_radii
        #------
        ligand_fragm_point_list = []
        for ligand_num in self.ligand_list:
            tmp_coord = np.array([0.0, 0.0, 0.0], dtype="float64")
            if ligand_num == self.tgt_ligand:
                continue

            for lig_atom in ligand_num:
                tmp_coord += self.complex_coord[lig_atom]
            tmp_coord /= len(ligand_num)
            ligand_fragm_point_list.append(tmp_coord)
        
        #print("ligand_fragm_point_list:", ligand_fragm_point_list)
        #------
        
        tgt_removed_z_vec_complex_coord = []
        tgt_removed_z_vec_complex_element = []
        for i in range(len(z_vec_metal_centerized_complex_coord)):
            if not i in self.tgt_ligand: 
                tgt_removed_z_vec_complex_coord.append(z_vec_metal_centerized_complex_coord[i])
                tgt_removed_z_vec_complex_element.append(self.complex_element_list[i])

        tgt_removed_z_vec_complex_coord = np.array(tgt_removed_z_vec_complex_coord, dtype="float64") 
        #------
    

        z_vec_donor_centerized_sub_ligand_coord = self.opt_place(tgt_removed_z_vec_complex_coord, init_distance, donor_centerized_sub_ligand_coord)

        for i in range(len(tgt_removed_z_vec_complex_coord)):
            print(tgt_removed_z_vec_complex_element[i]+"  {0:.12f}   {1:.12f}   {2:.12f}".format(tgt_removed_z_vec_complex_coord[i][0], tgt_removed_z_vec_complex_coord[i][1], tgt_removed_z_vec_complex_coord[i][2]))

        for i in range(len(z_vec_donor_centerized_sub_ligand_coord)):
            print(self.sub_ligand_element_list[i]+"  {0:.12f}   {1:.12f}   {2:.12f}".format(z_vec_donor_centerized_sub_ligand_coord[i][0], z_vec_donor_centerized_sub_ligand_coord[i][1], z_vec_donor_centerized_sub_ligand_coord[i][2]))

        
        self.substituted_coord = np.concatenate((tgt_removed_z_vec_complex_coord, z_vec_donor_centerized_sub_ligand_coord))
        self.substituted_element_list = tgt_removed_z_vec_complex_element + self.sub_ligand_element_list
        
        self.broken_flag = self.check_broken_struct(self.substituted_coord, self.substituted_element_list)
        
        return
    
    def run_bidentate_lig(self):
        donor_centerized_complex_coord = self.complex_coord - self.input_donor_center_coord
        
        dcenter_2_d_vec = self.input_donor_coord_list[0] - self.input_donor_center_coord
        donor_centerized_sub_ligand_coord = self.sub_ligand_coord - self.sub_donor_center_coord
      
        sub_dcenter_2_d_vec = self.sub_donor_coord_list[0] - self.sub_donor_center_coord
        d2d_z_rot_mat = self.make_rotmat_vec2z(dcenter_2_d_vec)
        sub_d2d_z_rot_mat = self.make_rotmat_vec2z(sub_dcenter_2_d_vec)

        z_vec_dcenter_2_d_centrized_complex_coord = np.dot(d2d_z_rot_mat, donor_centerized_complex_coord.T).T
        z_vec_sub_dcenter_2_d_centrized_sub_ligand_coord = np.dot(sub_d2d_z_rot_mat, donor_centerized_sub_ligand_coord.T).T

        tgt_removed_z_vec_complex_coord = []
        tgt_removed_z_vec_complex_element = []
        for i in range(len(z_vec_dcenter_2_d_centrized_complex_coord)):
            if not i in self.tgt_ligand: 
                tgt_removed_z_vec_complex_coord.append(z_vec_dcenter_2_d_centrized_complex_coord[i])
                tgt_removed_z_vec_complex_element.append(self.complex_element_list[i])

        tgt_removed_z_vec_complex_coord = np.array(tgt_removed_z_vec_complex_coord, dtype="float64") 


        z_vec_sub_dcenter_2_d_centrized_sub_ligand_coord = self.opt_place_bidentate(tgt_removed_z_vec_complex_coord, z_vec_sub_dcenter_2_d_centrized_sub_ligand_coord)


        for i in range(len(tgt_removed_z_vec_complex_coord)):
            print(tgt_removed_z_vec_complex_element[i]+"  {0:.12f}   {1:.12f}   {2:.12f}".format(tgt_removed_z_vec_complex_coord[i][0], tgt_removed_z_vec_complex_coord[i][1], tgt_removed_z_vec_complex_coord[i][2]))

        for i in range(len(z_vec_sub_dcenter_2_d_centrized_sub_ligand_coord)):
            print(self.sub_ligand_element_list[i]+"  {0:.12f}   {1:.12f}   {2:.12f}".format(z_vec_sub_dcenter_2_d_centrized_sub_ligand_coord[i][0], z_vec_sub_dcenter_2_d_centrized_sub_ligand_coord[i][1], z_vec_sub_dcenter_2_d_centrized_sub_ligand_coord[i][2]))
      
        self.substituted_coord = np.concatenate((tgt_removed_z_vec_complex_coord, z_vec_sub_dcenter_2_d_centrized_sub_ligand_coord))
        self.substituted_element_list = tgt_removed_z_vec_complex_element + self.sub_ligand_element_list
        self.broken_flag = self.check_broken_struct(self.substituted_coord, self.substituted_element_list)
        
        return
    

    def check_broken_struct(self, geometry, element_list, threshold_scaling=0.40):
        diff = geometry[:, np.newaxis, :] - geometry[np.newaxis, :, :]
        distances = np.linalg.norm(diff, axis=2)
        
        covalent_radiis = np.array([covalent_radii_lib(element) for element in element_list])
        thresholds = threshold_scaling * (covalent_radiis[:, np.newaxis] + covalent_radiis[np.newaxis, :])

        
        np.fill_diagonal(distances, np.inf)
        broken_flag = np.any(distances < thresholds)
        if broken_flag:
            print("This structure may be broken...")
        
        return broken_flag
    

    def run(self):
        self.complex_coord, self.complex_element_list = read_xyz(self.complex_abs_path)
        self.sub_ligand_coord, self.sub_ligand_element_list = read_xyz(self.sub_ligand_abs_path)

        self.metal_center_coord = np.array([0.0, 0.0, 0.0], dtype="float64")
        self.input_donor_center_coord = np.array([0.0, 0.0, 0.0], dtype="float64")
        self.sub_donor_center_coord = np.array([0.0, 0.0, 0.0], dtype="float64")
        self.input_donor_coord_list = []
        self.sub_donor_coord_list = []


        for i in range(len(self.metal_atom)):
            self.metal_center_coord += self.complex_coord[self.metal_atom[i] - 1]
        self.metal_center_coord /= len(self.metal_atom)

        for i in range(len(self.donor_atom)):
            tmp_coord = np.array([0.0, 0.0, 0.0], dtype="float64")
            for j in range(len(self.donor_atom[i])):
                tmp_coord += self.complex_coord[self.donor_atom[i][j] - 1]
            tmp_coord /= len(self.donor_atom[i])
            self.input_donor_coord_list.append(tmp_coord)

            self.input_donor_center_coord += tmp_coord
        
        self.input_donor_center_coord /= len(self.donor_atom)
        self.input_donor_coord_list = np.array(self.input_donor_coord_list, dtype="float64")


        for i in range(len(self.sub_donor_atom)):
            tmp_coord = np.array([0.0, 0.0, 0.0], dtype="float64")
            for j in range(len(self.sub_donor_atom[i])):
                tmp_coord += self.sub_ligand_coord[self.sub_donor_atom[i][j] - 1]
            tmp_coord /= len(self.sub_donor_atom[i])
            self.sub_donor_coord_list.append(tmp_coord)

            self.sub_donor_center_coord += tmp_coord
        
        self.sub_donor_center_coord /= len(self.sub_donor_atom)
        self.sub_donor_coord_list = np.array(self.sub_donor_coord_list, dtype="float64")


        self.ligand_list, self.tgt_ligand = make_lig_fragment_list(self.complex_coord, self.complex_element_list, self.metal_atom, self.donor_atom, self.covalent_radii_threshold_scale)



        if self.ndonor == 1:
            self.run_monodentate_lig()
        elif self.ndonor == 2:
            self.run_bidentate_lig()
        #elif self.ndonor == 3:
        #    self.run_tridentate_lig()
        else:
            print("The number of donor atoms must be 1 or 2.")
            sys.exit()

        save_xyz(self.complex_abs_path, self.substituted_coord, self.substituted_element_list, os.path.splitext(os.path.basename(self.sub_ligand_abs_path))[0])

        return





if __name__ == '__main__':
    args = parser()
    #print(args)
    if len(args["input_donor_atom"]) != len(args["sub_donor_atom"]):
        print("The number of donor atoms and the number of substituting atoms must be the same.")
        sys.exit()
    
    if len(args["input_donor_atom"]) < 4:
        sub = SubstituteLigand(**args)
    else:
        print("The number of donor atoms must be 1 or 2.")
        sys.exit()
    
    sub.run()


