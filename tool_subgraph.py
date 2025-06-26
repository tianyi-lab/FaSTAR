import networkx as nx
import re
from astar_search import extract_metadata_from_node

def parse_subtask(subtask_name):
    pattern = r"^(.*?)\s*(?:\((.*?)\))?\s*\((.*?)\)\s*$"
    match = re.match(pattern, subtask_name.strip())
    if not match:
        return {
            "subtask_type": subtask_name.strip(),
            "from_object": "",
            "target_object": "",
            "subtask_number": ""
        }

    subtask_type = match.group(1).strip()
    object_info = match.group(2)
    subtask_number = match.group(3).strip()
    from_obj, to_obj = "", ""

    if object_info:
        object_info = object_info.strip()
        if "->" in object_info:
            parts = object_info.split("->", 1)
            from_obj = parts[0].strip()
            to_obj = parts[1].strip()
        else:
            from_obj = object_info
    return {
        "subtask_type": subtask_type,
        "from_object": from_obj,
        "target_object": to_obj,
        "subtask_number": subtask_number
    }

def get_models_for_subtask(mdt, subtask_str):
    matches = []
    lower_sub = subtask_str.lower()
    for entry in mdt:
        tasks_supported = [t.strip().lower() for t in entry.get("Tasks Supported", "").split(",")]
        for t in tasks_supported:
            if t in lower_sub:
                matches.append(entry["Model"])
                break
    return matches

def backtrack_dependencies(tool_dependency_graph, models):
    required = set()
    for m in models:
        if m in tool_dependency_graph:
            ancestors = nx.ancestors(tool_dependency_graph, m)
            required.update(ancestors)
            required.add(m)
    return tool_dependency_graph.subgraph(required).copy()

def build_subgraph_for_subtask(
    tool_dependency_graph: nx.DiGraph,
    subtask_info: dict,
    final_models: list,
    parent_leaf_nodes: list,
    is_first_subtask: bool,
    global_graph: nx.DiGraph
):
    def rename_node(original_tool_name):
        if original_tool_name == "Input Image":
            return "Input Image"
        stype = subtask_info["subtask_type"]
        subtask_number = subtask_info["subtask_number"]
        from_obj = subtask_info["from_object"]
        to_obj = subtask_info["target_object"]
        if from_obj and to_obj:
            object_info = f"{from_obj} -> {to_obj}"
        elif from_obj:
            object_info = from_obj
        else:
            object_info = ""
        return f"{original_tool_name} ({stype} ({object_info})({subtask_number}))"

    sg = backtrack_dependencies(tool_dependency_graph, final_models)
    if not is_first_subtask and "Input Image" in sg.nodes():
        sg.remove_node("Input Image")

    rename_map = {node: rename_node(node) for node in sg.nodes()}
    sg_renamed = nx.DiGraph()
    for n in sg.nodes():
        sg_renamed.add_node(rename_map[n])
    for (u, v) in sg.edges():
        sg_renamed.add_edge(rename_map[u], rename_map[v])

    root_nodes = [n for n in sg_renamed if sg_renamed.in_degree(n) == 0]
    if is_first_subtask:
        for r in root_nodes:
            if r != "Input Image":
                global_graph.add_node(r)
                global_graph.add_edge("Input Image", r)
    else:
        for r in root_nodes:
            global_graph.add_node(r)
            for leaf in parent_leaf_nodes:
                global_graph.add_edge(leaf, r)

    for n in sg_renamed.nodes():
        global_graph.add_node(n)
    for (u, v) in sg_renamed.edges():
        global_graph.add_edge(u, v)

    leaves = []
    for fm in final_models:
        if fm in rename_map:
            renamed_fm = rename_map[fm]
            if sg_renamed.out_degree(renamed_fm) == 0:
                leaves.append(renamed_fm)
    return leaves

def build_tool_subgraph_from_subtask_chain(subtask_chain_json):

    tool_dependency_graph = nx.DiGraph()
   
    tool_dependency_graph.add_edge("Input Image", "YOLOv7")
    tool_dependency_graph.add_edge("Input Image", "GroundingDINO")
    tool_dependency_graph.add_edge("Input Image", "GoogleCloudVision")
    tool_dependency_graph.add_edge("Input Image", "RealESRGAN")
    tool_dependency_graph.add_edge("Input Image", "MagicBrush")
    tool_dependency_graph.add_edge("Input Image", "MIDAS")
    tool_dependency_graph.add_edge("Input Image", "CRAFT")
    tool_dependency_graph.add_edge("Input Image", "pix2pix")
    tool_dependency_graph.add_edge("Input Image", "StabilitySearchRecolor")
    tool_dependency_graph.add_edge("Input Image", "StabilityOutpaint")
    tool_dependency_graph.add_edge("Input Image", "StabilityRemoveBG")
    tool_dependency_graph.add_edge("Input Image", "Stability3")
    tool_dependency_graph.add_edge("Input Image", "Gpt4o_1")
    tool_dependency_graph.add_edge("Input Image", "DeblurGAN")

    tool_dependency_graph.add_edge("YOLOv7", "SAM")
    tool_dependency_graph.add_edge("GroundingDINO", "SAM")

    tool_dependency_graph.add_edge("SAM", "DalleImage")
    tool_dependency_graph.add_edge("SAM", "StabilityInpaint")
    tool_dependency_graph.add_edge("SAM", "StabilityErase")

    tool_dependency_graph.add_edge("CRAFT", "DeepFont")
    tool_dependency_graph.add_edge("CRAFT", "EasyOCR")

    tool_dependency_graph.add_edge("DeepFont", "Gpt4o_2")
    tool_dependency_graph.add_edge("EasyOCR", "Gpt4o_2")

    tool_dependency_graph.add_edge("EasyOCR", "CLIP")

    tool_dependency_graph.add_edge("Gpt4o_2", "TextRedaction")
    tool_dependency_graph.add_edge("Gpt4o_2", "TextWritingPillow1")
    tool_dependency_graph.add_edge("Gpt4o_2", "TextRemovalPainting")
    tool_dependency_graph.add_edge("Gpt4o_2", "DalleText")
    tool_dependency_graph.add_edge("Gpt4o_2", "StabilityEraseText")

    tool_dependency_graph.add_edge("DalleText", "TextWritingPillow2")
    tool_dependency_graph.add_edge("StabilityEraseText", "TextWritingPillow2")
    tool_dependency_graph.add_edge("TextRemovalPainting", "TextWritingPillow2")

    mdt = [
        {"Model": "YOLOv7", "Tasks Supported": "Object Detection"},
        {"Model": "GroundingDINO", "Tasks Supported": "Object Detection"},
        {"Model": "SAM", "Tasks Supported": "Object Segmentation"},
        {"Model": "DalleImage", "Tasks Supported": "Object Replacement"},
        {"Model": "DalleText", "Tasks Supported": "Text Removal"},
        {"Model": "StabilityInpaint", "Tasks Supported": "Object Replacement, Object Recoloration, Object Removal"},
        {"Model": "StabilitySearchRecolor", "Tasks Supported": "Object Recoloration"},
        {"Model": "StabilityOutpaint", "Tasks Supported": "Outpainting"},
        {"Model": "StabilityRemoveBG", "Tasks Supported": "Background Removal"},
        {"Model": "StabilityErase", "Tasks Supported": "Object Removal"},
        {"Model": "StabilityEraseText", "Tasks Supported": "Text Removal"},
        {"Model": "Stability3", "Tasks Supported": "Changing Scenery"},
        {"Model": "TextRemovalPainting", "Tasks Supported": "Text Removal"},
        {"Model": "DeblurGAN", "Tasks Supported": "Image Deblurring"},
        {"Model": "GPT4o_1", "Tasks Supported": "Image Captioning"},
        {"Model": "GPT4o_2", "Tasks Supported": "Question Answering based on text, Sentiment Analysis"},
        {"Model": "GoogleCloudVision", "Tasks Supported": "Landmark Detection"},
        {"Model": "CRAFT", "Tasks Supported": "Text Detection"},
        {"Model": "CLIP", "Tasks Supported": "Caption Consistency Check "},
        {"Model": "DeepFont", "Tasks Supported": "Text Style Detection"},
        {"Model": "EasyOCR", "Tasks Supported": "Text Extraction"},
        {"Model": "MagicBrush", "Tasks Supported": "Object Addition"},
        {"Model": "pix2pix", "Tasks Supported": "Changing Scenery"},
        {"Model": "RealESRGAN", "Tasks Supported": "Image Upscaling"},
        {"Model": "TextWritingPillow1", "Tasks Supported": "Text Addition"},
        {"Model": "TextWritingPillow2", "Tasks Supported": "Text Replacement, Keyword Highlighting"},
        {"Model": "TextRedaction", "Tasks Supported": "Text Redaction"},
        {"Model": "MIDAS", "Tasks Supported": "Depth Estimation"}
    ]

    global_graph = nx.DiGraph()
    global_graph.add_node("Input Image")
    subtask_leaf_map = {}
    subtask_list = subtask_chain_json.get("subtask_chain", [])
    
    for node in subtask_list:
        sname = node.get("subtask")
        if not sname:
            continue
        parents = node.get("parent", [])
        subtask_info = parse_subtask(sname)
        models_for_subtask = get_models_for_subtask(mdt, subtask_info["subtask_type"])
        parent_leaf_nodes = []
        for p in parents:
            parent_leaf_nodes.extend(subtask_leaf_map.get(p, []))
        is_first = (len(parents) == 0)
        new_leaves = build_subgraph_for_subtask(
            tool_dependency_graph,
            subtask_info,
            models_for_subtask,
            parent_leaf_nodes,
            is_first,
            global_graph
        )
        subtask_leaf_map[sname] = new_leaves

    adjacency_dict = {node: list(global_graph.successors(node)) for node in global_graph.nodes()}
    return adjacency_dict

def main():
    # Example usage
    subtask_chain_json = {
        "subtask_chain": [
            {"subtask": "Object Detection (YOLOv7) (1)"},
            {"subtask": "Object Segmentation (SAM) (2)", "parent": ["Object Detection (YOLOv7) (1)"]},
            {"subtask": "Object Replacement (DalleImage) (3)", "parent": ["Object Segmentation (SAM) (2)"]}
        ]
    }
    adjacency_dict = build_tool_subgraph_from_subtask_chain(subtask_chain_json)
    print(adjacency_dict)

# # Uncomment the lines below to run the main function when this script is executed
# if __name__ == "__main__":
#     main()
