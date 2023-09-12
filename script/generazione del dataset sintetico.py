# caricamento del dataset e rilevamento dei metadati
from sdv.datasets.local import load_csvs

# datasets è un dizionario che contiene tutti i file CSV trovati nella cartella 'NSL-KDD'.
# la chiave è il nome del file (senza il suffisso .csv) e il valore è un DataFrame pandas contenente i dati.
datasets = load_csvs(folder_name='NSL-KDD/')

#KDD_table contiene i dati del file CSV 'KDDTrain+' presente all'interno della cartella 'NSL-KDD'
KDD_table = datasets['KDDTrain+']

from sdv.metadata import SingleTableMetadata

# i metadati sono rilevati automaticamente dai dati contenuti nel file CSV
metadata = SingleTableMetadata()
metadata.detect_from_csv(filepath='NSL-KDD/KDDTrain+.csv')

# aggiornamento di alcune colonne categoriche che erano state considerate numeriche
metadata.update_column(
    column_name="'land'",
    sdtype='categorical')

metadata.update_column(
    column_name="'logged_in'",
    sdtype='categorical')

metadata.update_column(
    column_name="'is_host_login'",
    sdtype='categorical')

metadata.update_column(
    column_name="'is_guest_login'",
    sdtype='categorical')

metadata.set_primary_key(column_name='id')
metadata.save_to_json(filepath='metadati.json')



# Step 1: Creare il sintetizzatore
from sdv.single_table import CTGANSynthesizer

# synthesizer è un oggetto CTGANSynthesizer con caratteristiche personalizzate
synthesizer = CTGANSynthesizer(
    metadata, # required
    generator_lr=0.0002,
    discriminator_lr=0.0002,
    batch_size=500,
    discriminator_steps=5,
    epochs=100,
    verbose=True
)

# Step 2: Addestrare il sintetizzatore sulla base dei dati reali contenuti in KDD_table
synthesizer.fit(KDD_table)

# Step 3: Generazione dei dati sintetici
# Dopo l'addestramento il sintetizzatore può essere usato per generare nuovi dati sintetici. 
# La quantità di dati da generare è stata impostata pari a quella del dataset originale
synthetic_data = synthesizer.sample(len(KDD_table))

# salvataggio del dataset sintetico in un file CSV denominato 'synthetic_data'
synthetic_data.to_csv('synthetic_data.csv', index=False)
