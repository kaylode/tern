import matplotlib.pyplot as plt
from PIL import Image

def draw_retrieval_results(queries, top_k_relevant, save_filename=None):
    ax = []
    post_list = queries + top_k_relevant
    fig=plt.figure(figsize=(10, 10))
    rows = len(top_k_relevant)
    columns = len(queries)
    for i in range(columns*rows):
        img = Image.open(post_list[i][0])
        ax.append(fig.add_subplot(rows, columns, i+1) )
        ax[-1].set_title(post_list[i][1])  # set title
        plt.imshow(img)
        plt.tight_layout()
        plt.axis('off')
    if save_filename is not None:
        plt.savefig(save_filename)
    return fig
