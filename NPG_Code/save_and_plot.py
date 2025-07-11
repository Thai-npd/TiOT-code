import numpy as np
import matplotlib.pyplot as plt
import os
from openpyxl import load_workbook, Workbook
import openpyxl
import seaborn as sns
import matplotlib.patches as mpatches

def save(data, name_instance,name, saving_obj):
    if saving_obj == 1:
        saved_obj = "data"
    elif saving_obj == 2:
        saved_obj = "results"
    else:
        print("Invalid saving_obj in save_result. Saving objective have to be either 1(for data) or 2(for result). Program exited.")
        exit()
    folder_path = os.path.join(saved_obj, name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    filename = os.path.join(folder_path, name_instance)
    np.save(filename, data)

def load(name_instance,name, saving_obj):
    if saving_obj == 1:
        saved_obj = "data"
    elif saving_obj == 2:
        saved_obj = "results"
    else:
        print("Invalid saving_obj in save_result. Saving objective have to be either 1(for data) or 2(for result). Program exited.")
        exit()
    folder_path = os.path.join(saved_obj, name)
    filename = os.path.join(folder_path, name_instance) + ".npy"
    loaded_data = np.load(filename, allow_pickle=True)
    return loaded_data.item()

def make_all_plots(results, name_instance,name,save = True, plot_mse = False, plot = True):
    plot_res(results, "res", name_instance,name,save = save, plot=plot)
    plot_res(results, "obj", name_instance,name,save = save, plot=plot)
    plot_res(results, "obj", name_instance, name ,save = save,over_time=True, plot=plot)
    plot_res(results, "steps", name_instance, name,save = save, plot=plot)
    boxplot(results, name_instance, name,save = save, plot=plot)
    if plot_mse == True:
        plot_res(results, "mse", name_instance, name,save = save, plot=plot)


def find_min_value(results):
    f_min = np.inf
    for history_values in results.values():
        f_min = min(f_min, min(history_values["obj"]))
    return f_min

def assign_color(alg_names):
    color_map = {
    'red': [ "#fc8d59", '#d73027', "#92580b"],    # For 'NPG' group #fc8d59
    'green': ["#5dc404", "#045D2C", "#09EE70ED", "#022211EC"],             # For 'LS' group
    'blue': ["#6d36ab", '#2171b5']                # AdPG group
}
    assigned_colors = []

    # Track usage index for each group
    color_index = {'red': 0, 'green': 0, 'blue': 0}

    for name in alg_names:
        if 'NPG' in name:
            group = 'red'
        elif 'LS' in name:
            group = 'green'
        else:
            group = 'blue'
        
        # Pick next color in group's list (cycle if needed)
        group_colors = color_map[group]
        index = color_index[group] % len(group_colors)
        assigned_colors.append(group_colors[index])
        color_index[group] += 1
    return assigned_colors

def plot_res(results, key, name_instance,name ,over_time = False, with_fopt = True, save = True, plot = True):
    sns.set(style="whitegrid", context="paper")
    plt.figure()
    name_image = name_instance + "_"+ key
    fontsize = 14
    if over_time == True: 
        name_image += "_over_time"
        plt.xlabel('Time(s)', fontsize=fontsize)
    else:
        plt.xlabel('Iterations', fontsize=fontsize)
    if key == "obj":
        plt.ylabel('Objective', fontsize=fontsize)
    elif key == "res":
        plt.ylabel('Residual', fontsize=fontsize)
    elif key == "grad":
        plt.ylabel("Norm of gradient", fontsize=fontsize)
    elif key == "steps":
        plt.ylabel("Stepsize", fontsize=fontsize)
    elif key == "mse":
        plt.ylabel("Mean squared error", fontsize=fontsize)
    name_image += ".pdf"
    markers = ['o', '*', 'D', '^', 'v', '<', '>', 'x', '+', '.']
    if key == "obj" and with_fopt:
        opt = find_min_value(results) 
    else:
        opt = 0.0
    i = 0
    numerical_stabalizer = 0 
    algs = results.keys()
    colors = assign_color(algs)
    for alg, history_values in results.items():
        N = len(history_values[key])
        values = np.array(history_values[key]) - opt + numerical_stabalizer
        if with_fopt: 
            mask = values > 0
        else:
            mask = [True for i in range(N)]
        printed_values = values[mask]
        if over_time == True:
            time_axis = np.array(history_values["time"])[mask]
            if len(set(time_axis)) != len(time_axis):
                print("Couldn't plot objective over time due to too small time step.")
                return None
            plt.plot(time_axis, printed_values, label = alg, marker = markers[i % 10], markevery = int(N / 10) +1, color = colors[i], linewidth=1.5 ) 
        else:
            plt.plot(printed_values, label=alg, marker = markers[i % 10], markevery = int(N / 10)+1, color = colors[i], linewidth=1.5 )
        i += 1
    #plt.grid(True)
    plt.yscale("log")
    #plt.title(name_instance)
    plt.legend()
    plt.tight_layout()
    if save:
        folder_path = os.path.join("plots", name, name_instance)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        plt.savefig(os.path.join(folder_path, name_image), dpi=300)
    if plot:
        plt.show()
    plt.close()

def boxplot(results, name_instance, name , save = True, plot = True):
    name_image = name_instance + "_" + 'stepsize_boxplot'
    fontsize = 16
    name_image += ".pdf"
    
    data_values = list( d['steps'] for d in results.values())
    labels = list(results.keys())
    colors = assign_color(labels)  # Customize colors here

    # Create box plot
    sns.set(style="whitegrid", context="paper")
    fig, ax = plt.subplots(figsize=(12,8))
    box = ax.boxplot(data_values, patch_artist=True, showfliers=False, medianprops=dict(color='black', linewidth=1.5))  # patch_artist=True enables fill color

    # Set colors
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)

    # Customize plot
    #ax.set_xticks(range(1, len(labels) + 1))
    #ax.set_xticklabels(labels, fontsize = fontsize - 3)
    ax.set_xticks([])
    ax.set_ylabel('Stepsize', fontsize = fontsize)
    ax.set_xlabel('Algorithms', fontsize = fontsize)
    handles = [mpatches.Patch(color=c, label=l) for c, l in zip(colors, labels)]
    ax.legend(handles=handles, loc='upper left', fontsize=fontsize - 4)
    #ax.legend(handles=handles, loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=9)
    if save:
        folder_path = os.path.join("plots", name, name_instance)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        plt.savefig(os.path.join(folder_path, name_image), dpi=300)
    if plot:
        plt.show()
    plt.close()

def create_report(results, name, position, parameters , name_workbook, with_fopt = True, avg_ite = None):
    if os.path.exists(name_workbook):
        workbook = load_workbook(name_workbook)
    else:
        workbook = Workbook()
        for sheet_name in workbook.sheetnames:
            sheet = workbook[sheet_name]
            workbook.remove(sheet)
    if name in workbook.sheetnames:
        worksheet = workbook[name]
    else:
        worksheet = workbook.create_sheet(name)

    row, col = position[0], position[1]

    if with_fopt:
        opt = find_min_value(results)
    else:
        opt = 0
    worksheet.cell(row=row, column=col, value=parameters["size"] + " " + parameters["additional_info"])
    row += 1
    worksheet.cell(row=row, column=col, value="data")
    worksheet.cell(row=row + 1, column=col, value=parameters["seed"])
    worksheet.cell(row=row, column=col + 1, value="k")
    worksheet.cell(row=row, column=col + 2, value="Residual")
    worksheet.cell(row=row, column=col + 3, value="Objective")
    worksheet.cell(row=row, column=col + 4, value="Time")
    if "lasso" in name: worksheet.cell(row=row, column=col + 5, value="Mse")        # Only report mse for the lasso problem
    row += 1

    for k, history_values in results.items():
        index = len(history_values["res"]) - 1
        if avg_ite != None: 
            worksheet.cell(row=row, column=col + 1, value=avg_ite[k])
        else: 
            worksheet.cell(row=row, column=col + 1, value=index + 1) 
        worksheet.cell(row=row, column=col + 2, value=history_values["res"][index])
        worksheet.cell(row=row, column=col + 3, value=history_values["obj"][index] - opt)
        worksheet.cell(row=row, column=col + 4, value=history_values["time"][index])
        if "lasso" in name: 
            worksheet.cell(row=row, column=col + 5, value=history_values["mse"][index])
            worksheet.cell(row=row, column=col + 6, value=k)
        else:
            worksheet.cell(row=row, column=col + 5, value=k)

        row += 1
    workbook.save(name_workbook)

def format_latex(results, name, with_fopt = True, avg_ite = None):
    output = []
    if with_fopt:
        opt = find_min_value(results)
    else:
        opt = 0
    excel_file_path = 'latex_support.xlsx'
    if not os.path.exists(excel_file_path):
        wb = openpyxl.Workbook()
        ws = wb.active
        wb.save(excel_file_path)
    else:
        wb = openpyxl.load_workbook(excel_file_path)
        if name in wb.sheetnames:
            ws = wb[name]
        else:
            ws = wb.create_sheet(name)  

    row = ws.max_row + 1
    col = 1
    for k, history_values in results.items():
        index = len(history_values["res"]) - 1
        if avg_ite != None: 
            ws.cell(row=row, column=col , value=avg_ite[k])
        else: 
            ws.cell(row=row, column=col, value=index + 1) 
        ws.cell(row=row + 1, column=col, value=history_values["res"][index])
        ws.cell(row=row + 2, column=col, value=history_values["obj"][index] - opt)
        ws.cell(row=row + 3, column=col, value=history_values["time"][index])
        if "lasso" in name: 
            ws.cell(row=row+4, column=col, value=history_values["mse"][index])
        col += 1

    wb.save(excel_file_path)