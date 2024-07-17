'''
Simulate and plot phylogenetic trees using a birth-death process.
'''
import ete3
import ngesh

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg



#-------------------------------------------------------------------------------
# Parameters
#-------------------------------------------------------------------------------

# Tree generation
birth_rates = [ 0.25, 0.75, 1.5 ]
death_rates = [ 0.25 ]
min_leaves = 20
#-------------------------------------------------------------------------------



def main():

    # Initialize
    seed = 5
    n_birth = len(birth_rates)
    n_death = len(death_rates)

    # Loop through birth/death parameters:
    for i, birth in enumerate(birth_rates):
        for j, death in enumerate(death_rates):
        
            print( "... generating tree with birth = ", birth, " and death = ", death )
            try:
                tree = ngesh.random_tree.gen_tree( min_leaves = min_leaves,
                                                   birth      = birth,
                                                   death      = death,
                                                   seed       = seed
                                                  )
                title = "Birth-death tree with\n" +\
                "λ=" + str(birth) + " and μ=" + str(death)+ "\n"                                 
                filename = "tree--birth-" + str(birth) + "-death-" + str(death) 
                filename = filename.replace(".", "_") + ".png"
                
                ts = ete3.TreeStyle()
                ts.title.add_face( ete3.TextFace(title, fsize=9), column=0 )
                ts.show_leaf_name     = False
                ts.show_branch_length = False
                if birth==0.25:
                    ts.scale = 10
                    ts.branch_vertical_margin = 11
                elif birth==0.75:
                    ts.scale = 71.4
                    ts.branch_vertical_margin = 30
                elif birth==1.5:
                    ts.scale = 125  # 130
                    ts.branch_vertical_margin = 35

                #ts.show_scale = False
                
                # Set the width of the branches in the tree visualization  
                node_style = ete3.NodeStyle()  
                node_style["hz_line_width"] = 2  # Set the width of horizontal branches  
                node_style["vt_line_width"] = 3  # Set the width of vertical branches  
                for n in tree.traverse():  
                    n.set_style(node_style)
                
                tree.render(filename, tree_style=ts)
                #tree.show(tree_style=ts)
                img = mpimg.imread(filename)
                imgplot = plt.imshow(img)
                plt.axis('off')
                plt.show()

            except RuntimeError as e:
                print( "... ERROR: generating tree with birth = ", birth, 
                       " and death = ", death, " --- ", e )
            seed += 1
        
    return
    
    
if __name__ == "__main__":
    main()
