


###########################
# ----- Extra Tools ----- #
###########################


def epoch_converter(epoch_time, h=False):
    """
    Converts epoch time to a formatted string in Spanish time.
    
    Parameters:
        epoch_time (float): The epoch time to convert.
        h (bool): If True, include the hour and minute in the format. Defaults to False.
        
    Returns:
        str: The formatted date string.
    """
    # Set locale to Spanish
    locale.setlocale(locale.LC_TIME, 'es_ES.UTF-8')           #adjust to your system locale if needed
    # Convert epoch time to a datetime object
    dt_object = datetime.fromtimestamp(epoch_time)
    # Use different formats based on the value of h
    if h:
        # formatted_date = dt_object.strftime('%H:%M')
        formatted_date = dt_object.strftime('%d/%m - %H:%M')  #day/month - hour:minute
    else:
        formatted_date = dt_object.strftime('%d/%m')          #day/month
    
    return formatted_date