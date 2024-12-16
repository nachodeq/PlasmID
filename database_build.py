import os
from pymongo import MongoClient, InsertOne
import pandas as pd
from Bio import SeqIO
import json
import logging
import csv

ENVIRONMENT_CATEGORIES = {
    "Water": ["water", "marine", "river", "lake", "seawater", "ocean"],
    "Soil": ["soil", "ground", "earth", "compost", "sediment"],
    "Plant": ["plant", "tree", "leaf", "root", "phyllosphere"],
    "Animal Host": ["human", "animal", "gut", "intestine", "feces", "skin", "blood", "oral"],
    "Industrial/Lab": ["lab", "waste", "industrial", "bioreactor", "sewage"],
    "Unknown": ["unknown", "n/a", "none", "not available"]
}

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Parsing Functions
def parse_fasta(fasta_folder):
    plasmid_sequences = {}
    for filename in os.listdir(fasta_folder):
        if filename.endswith('.fasta') or filename.endswith('.fa') or filename.endswith('.fna'):
            filepath = os.path.join(fasta_folder, filename)
            try:
                for record in SeqIO.parse(filepath, 'fasta'):
                    plasmid_id = os.path.splitext(filename)[0]
                    plasmid_sequences[plasmid_id] = str(record.seq)
            except Exception as e:
                logging.error(f"Error processing {filename}: {e}")
    logging.info(f"Total plasmid sequences parsed: {len(plasmid_sequences)}")
    return plasmid_sequences

def parse_bakta(bakta_folder):
    plasmid_genes = {}
    for folder_name in os.listdir(bakta_folder):
        folder_path = os.path.join(bakta_folder, folder_name)
        if os.path.isdir(folder_path):
            plasmid_id = folder_name.replace('_baktaresult', '')
            json_file = os.path.join(folder_path, f"{plasmid_id}.json")
            if os.path.exists(json_file):
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                        genes = [
                            {
                                'locus': feature.get('locus', ''),
                                'id': feature.get('id', ''),
                                'gene': feature.get('gene', ''),
                                'product': feature.get('product', ''),
                                'start': feature.get('start', 0),
                                'stop': feature.get('stop', 0),
                                'strand': feature.get('strand', ''),
                                'contig': feature.get('contig', ''),
                                'db_xrefs': feature.get('db_xrefs', []),
                                'nt_sequence': feature.get('nt', ''),
                                'aa_sequence': feature.get('aa', ''),
                            }
                            for feature in data.get('features', [])
                            if feature.get('type', '').lower() == 'cds'
                        ]
                        plasmid_genes[plasmid_id] = genes
                except json.JSONDecodeError:
                    logging.error(f"Error decoding JSON for {plasmid_id}")
    logging.info(f"Total plasmid genes parsed: {sum(len(genes) for genes in plasmid_genes.values())}")
    return plasmid_genes

def parse_mobtyper(mobtyper_folder):
    plasmid_mobility = {}
    for filename in os.listdir(mobtyper_folder):
        if filename.endswith('_mobtyper.fasta'):
            filepath = os.path.join(mobtyper_folder, filename)
            plasmid_id = filename.replace('_mobtyper.fasta', '')
            try:
                with open(filepath, 'r') as f:
                    reader = csv.DictReader(f, delimiter='\t')
                    for row in reader:
                        plasmid_mobility[plasmid_id] = {
                            'mobility': row.get('predicted_mobility', '').strip(),
                            'replicon_type': row.get('rep_type(s)', '').strip()
                        }
            except Exception as e:
                logging.error(f"Error reading {filename}: {e}")
    logging.info(f"Total plasmid mobility data parsed: {len(plasmid_mobility)}")
    return plasmid_mobility

def parse_plsdb_metadata(metadata_file):
    metadata_df = pd.read_csv(metadata_file, sep='\t')
    
    # Check for required columns
    required_columns = [
        'NUCCORE_ACC',
        'BIOSAMPLE_IsolationSource',
        'TAXONOMY_genus',
        'TAXONOMY_species',
        'TAXONOMY_family',
        'ASSEMBLY_Status',
        'ASSEMBLY_ACC'
    ]
    missing_columns = [col for col in required_columns if col not in metadata_df.columns]
    if missing_columns:
        logging.error(f"The metadata file is missing required columns: {', '.join(missing_columns)}")
        # Continue processing but be aware that some data may be missing
        # You can choose to return None or handle this as per your requirement

    # Lowercase the 'BIOSAMPLE_IsolationSource' for categorization
    metadata_df['IsolationSource_lower'] = metadata_df['BIOSAMPLE_IsolationSource'].astype(str).str.lower()
    
    # Categorize environments
    def categorize(isolation_source):
        for category, keywords in ENVIRONMENT_CATEGORIES.items():
            if any(keyword in isolation_source for keyword in keywords):
                return category
        return "Other"
    
    metadata_df['Categorized_Environment'] = metadata_df['IsolationSource_lower'].apply(categorize)
    metadata_df.drop(columns=['IsolationSource_lower'], inplace=True)
    
    # Log the columns present
    logging.info(f"Columns in metadata_df: {metadata_df.columns.tolist()}")
    return metadata_df

def parse_resfinder_tab(resfinder_tab_file):
    resistance_genes = {}
    try:
        resfinder_data = pd.read_csv(resfinder_tab_file, sep='\t')
        resfinder_data.columns = resfinder_data.columns.str.strip()
        logging.info(f"Columns found in ResFinder data: {resfinder_data.columns.tolist()}")

        for _, row in resfinder_data.iterrows():
            try:
                resistance_gene = row['Resistance gene'].strip()
                contig_info = row['Contig'].split('|')
                plasmid_id = contig_info[0]
                gene_name = contig_info[1].split()[0] if len(contig_info) > 1 else ''
                query_id = f"{plasmid_id}|{gene_name}"

                resistance_genes[query_id] = {
                    'gene_name': resistance_gene,
                    'resistance_to': row['Phenotype'].split(', '),
                    'identity': float(row['Identity']),
                    'coverage': float(row['Coverage']),
                    'alignment_length': row['Alignment Length/Gene Length']
                }
            except ValueError as ve:
                logging.warning(f"Value error for row {row}: {ve}")
            except Exception as e:
                logging.error(f"Error processing row {row}: {e}")

        logging.info(f"Parsed {len(resistance_genes)} unique resistance genes from ResFinder data.")
    except Exception as e:
        logging.error(f"Error processing ResFinder data: {e}")

    return resistance_genes

def integrate_resistance_data(plasmid_genes, resistance_genes):
    integrated_count = 0
    unmatched_genes = 0

    for plasmid_id, genes in plasmid_genes.items():
        for gene in genes:
            query_id_key = f"{plasmid_id}|{gene['locus']}"

            if query_id_key in resistance_genes:
                gene['antibiotic_resistance'] = True
                gene['resistance_info'] = resistance_genes[query_id_key]
                integrated_count += 1
            else:
                gene['antibiotic_resistance'] = False
                unmatched_genes += 1

    logging.info(f"Total genes integrated with resistance data: {integrated_count}")
    logging.info(f"Total genes without matching resistance data: {unmatched_genes}")

    return plasmid_genes

# Insert environments
def insert_environments(metadata_df, db):
    environment_id_map = {}
    unique_environments = metadata_df['Categorized_Environment'].unique()
    environment_data_list = [{'name': env} for env in unique_environments]

    if environment_data_list:
        try:
            result = db.environments.insert_many(environment_data_list)
            for idx, environment in enumerate(unique_environments):
                environment_id_map[environment] = result.inserted_ids[idx]
            logging.info(f"Inserted {len(result.inserted_ids)} unique environments into the database.")
        except Exception as e:
            logging.error(f"Failed to insert environments: {e}")

    return environment_id_map

def insert_hosts(metadata_df, environment_id_map, db):
    host_id_map = {}
    host_data_list = []
    host_name_list = []

    if 'TAXONOMY_genus' not in metadata_df.columns or 'TAXONOMY_species' not in metadata_df.columns:
        logging.error("El archivo de metadata no contiene las columnas requeridas para los hosts.")
        return host_id_map

    # Agrupar por genus y species, y recopilar todos los environments asociados
    grouped_hosts = metadata_df.groupby(['TAXONOMY_genus', 'TAXONOMY_species'])['Categorized_Environment'].unique().reset_index()

    for _, row in grouped_hosts.iterrows():
        genus = row['TAXONOMY_genus']
        species_list = row['TAXONOMY_species'].split("_")
        species = ' '.join(species_list[:2])
        host_name = f"{genus} {species}"

        # Obtener los environment_ids únicos para este host
        environment_names = row['Categorized_Environment']
        environment_ids = [environment_id_map.get(env) for env in environment_names if environment_id_map.get(env)]

        if not environment_ids:
            logging.warning(f"No se encontraron environment_ids para el host {host_name}. Se omitirá este host.")
            continue

        # Verificar si el host ya existe
        existing_host = db.hosts.find_one({'genus': genus, 'species': species})
        if existing_host:
            # Actualizar el host existente añadiendo nuevos environment_ids
            try:
                result = db.hosts.update_one(
                    {'_id': existing_host['_id']},
                    {'$addToSet': {'environment_ids': {'$each': environment_ids}}}
                )
                host_id_map[host_name] = existing_host['_id']
                if result.modified_count > 0:
                    logging.info(f"Actualizado el host existente: {host_name} con nuevos environment_ids.")
            except Exception as e:
                logging.error(f"Error al actualizar el host {host_name}: {e}")
        else:
            # Insertar un nuevo host
            host_data = {
                'genus': genus,
                'species': species,
                'environment_ids': environment_ids
            }
            host_data_list.append(host_data)
            host_name_list.append(host_name)

    # Insertar los nuevos hosts
    if host_data_list:
        try:
            result = db.hosts.insert_many(host_data_list)
            for idx, inserted_id in enumerate(result.inserted_ids):
                host_id_map[host_name_list[idx]] = inserted_id
            logging.info(f"Insertados {len(result.inserted_ids)} nuevos hosts en la base de datos.")
        except Exception as e:
            logging.error(f"Fallo al insertar nuevos hosts: {e}")

    return host_id_map


def insert_plasmids(plasmid_sequences, plasmid_mobility, metadata_df, host_id_map, environment_id_map, db):
    plasmid_data_list = []
    plasmid_id_list = []
    plasmid_id_map = {}

    for plasmid_id, sequence in plasmid_sequences.items():
        plasmid_data = {
            'plasmid_id': plasmid_id,
            'sequence': sequence,
            'sequence_length': len(sequence),
            'mobility': plasmid_mobility.get(plasmid_id, {}).get('mobility'),
            'replicon_type': plasmid_mobility.get(plasmid_id, {}).get('replicon_type')
        }

        plasmid_metadata = metadata_df[metadata_df['NUCCORE_ACC'] == plasmid_id]
        if not plasmid_metadata.empty:
            environment_name = plasmid_metadata['Categorized_Environment'].values[0]
            host_genus = plasmid_metadata['TAXONOMY_genus'].values[0]
            species_list = plasmid_metadata['TAXONOMY_species'].values[0].split("_")
            host_species = ' '.join(species_list[:2])
            host_name = f"{host_genus} {host_species}"
            plasmid_data['environment_id'] = environment_id_map.get(environment_name)
            plasmid_data['host_id'] = host_id_map.get(host_name)

            # Include assembly information
            plasmid_data['assembly_status'] = plasmid_metadata['ASSEMBLY_Status'].values[0]
            plasmid_data['assembly_accession'] = plasmid_metadata['ASSEMBLY_ACC'].values[0]
        else:
            # Handle missing metadata
            plasmid_data['environment_id'] = None
            plasmid_data['host_id'] = None
            plasmid_data['assembly_status'] = None
            plasmid_data['assembly_accession'] = None

        plasmid_data_list.append(plasmid_data)
        plasmid_id_list.append(plasmid_id)

    if plasmid_data_list:
        try:
            result = db.plasmids.insert_many(plasmid_data_list)
            for idx, inserted_id in enumerate(result.inserted_ids):
                plasmid_id_map[plasmid_id_list[idx]] = inserted_id
            logging.info(f"Inserted {len(result.inserted_ids)} plasmids into the database.")
        except Exception as e:
            logging.error(f"Failed to insert plasmids: {e}")

    return plasmid_id_map

def insert_genes(plasmid_genes, plasmid_id_map, db):
    bulk_operations = []

    for plasmid_id, genes in plasmid_genes.items():
        plasmid_object_id = plasmid_id_map.get(plasmid_id)
        if not plasmid_object_id:
            continue  # Skip if plasmid was not inserted
        for gene in genes:
            gene_data = {
                'plasmid_id': plasmid_object_id,
                'locus': gene['locus'],
                'gene_id': gene['id'],
                'gene_name': gene['gene'],
                'product': gene['product'],
                'start': gene['start'],
                'stop': gene['stop'],
                'strand': gene['strand'],
                'contig': gene['contig'],
                'nt_sequence': gene['nt_sequence'],
                'aa_sequence': gene['aa_sequence'],
                'antibiotic_resistance': gene.get('antibiotic_resistance', False),
                'resistance_info': gene.get('resistance_info', {})
            }
            bulk_operations.append(InsertOne(gene_data))

    if bulk_operations:
        try:
            result = db.genes.bulk_write(bulk_operations)
            logging.info(f"Inserted {result.inserted_count} genes into the database.")
        except Exception as e:
            logging.error(f"Failed to insert genes: {e}")

# Main function
def main():
    # MongoDB setup
    username, password = 'XXXXXXX', 'XXXXXXX'
    client = MongoClient("mongodb://localhost:XXXXXXX")

    db = client['XXXXXXX']
    
    # Define file paths
    fasta_folder = 'XXXXXXX'
    bakta_folder = 'XXXXXXX'
    mobtyper_folder = 'XXXXXXX'
    metadata_file = 'XXXXXXX'
    resfinder_tab_file = 'XXXXXXX'
    
    # Parse files
    plasmid_sequences = parse_fasta(fasta_folder)
    plasmid_genes = parse_bakta(bakta_folder)
    plasmid_mobility = parse_mobtyper(mobtyper_folder)
    metadata_df = parse_plsdb_metadata(metadata_file)
    resistance_genes = parse_resfinder_tab(resfinder_tab_file)
    plasmid_genes = integrate_resistance_data(plasmid_genes, resistance_genes)

    # Insert environments, hosts, plasmids, and genes
    environment_id_map = insert_environments(metadata_df, db)
    if not environment_id_map:
        logging.error("No environment IDs found. Exiting.")
        return

    host_id_map = insert_hosts(metadata_df, environment_id_map, db)
    if not host_id_map:
        logging.error("No host IDs found. Exiting.")
        return

    plasmid_id_map = insert_plasmids(plasmid_sequences, plasmid_mobility, metadata_df, host_id_map, environment_id_map, db)
    if not plasmid_id_map:
        logging.error("No plasmid IDs found. Exiting.")
        return

    insert_genes(plasmid_genes, plasmid_id_map, db)
    
    logging.info("Data import completed.")

if __name__ == '__main__':
    main()

