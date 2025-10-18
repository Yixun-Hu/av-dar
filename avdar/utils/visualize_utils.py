from terminaltables import AsciiTable

def loss_table(loss_dict):
    """ Ascii table for logging losses """
    heading = [k for k in loss_dict.keys()]
    data = [f'{loss_dict[k]:4f}' for k in loss_dict.keys()]
    table_data = [heading, data]
    table = AsciiTable(table_data)
    return table.table