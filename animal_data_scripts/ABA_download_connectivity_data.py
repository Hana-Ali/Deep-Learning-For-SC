# Rembrandt Bakker, inspired by the Allen Institute's
# 'Structure Similarity Network' example at: 
# http://www.brain-map.org/api/examples/examples/structures/index.html
# 

import numpy
import json
import sys
import os
import urllib
import string
import csv

# Global variables defining the path to the API and IDs of the ontology, 
# structure graph, product, and plane of sectioning of interest.  This script
# also provides the option of only estimating connections between a specific
# set of structures using the StructureSets table.  Only the `TOP_N` most 
# correlated connections will be kept, approximately.

API_PATH = "http://api.brain-map.org/api/v2/data"
GRAPH_ID = 1
PLANE_ID = 1 # coronal
TOP_N = 2000

"""
DATA_SET_QUERY_URL = ("%s/SectionDataSet/query.json" +\
                          "?criteria=[failed$eqfalse]" +\
                          ",products[id$in5,6]" +\
                          ",[green_channel$eqrAAV]" +\
                          ",specimen(donor[transgenic_mouse_id$eqnull](age[days$ge74],[days$le80]))" +\
                          ",plane_of_section[id$eq%d]" +\
                          "&include=specimen(stereotaxic_injections(stereotaxic_injection_materials,stereotaxic_injection_coordinates,primary_injection_structure)),specimen(donor(age))") \
                          % (API_PATH, PLANE_ID)
"""
DATA_SET_QUERY_URL = ("%s/SectionDataSet/query.json" +\
                          "?criteria=[failed$eqfalse]" +\
                          ",products[id$in5]" +\
                          ",[green_channel$eqrAAV]" +\
                          ",specimen(donor[transgenic_mouse_id$eqnull])" +\
                          ",specimen(stereotaxic_injections(age[days$ge54],[days$le58]))" +\
                          ",plane_of_section[id$eq%d]" +\
                          "&include=specimen(stereotaxic_injections(age,stereotaxic_injection_materials,stereotaxic_injection_coordinates,primary_injection_structure)),specimen(donor(age))") \
                          % (API_PATH, PLANE_ID)

UNIONIZE_FMT = "%s/ProjectionStructureUnionize/query.json" +\
               "?criteria=[section_data_set_id$eq%d],[is_injection$eqfalse]" +\
               "&include=hemisphere"

STRUCTURES_URL = ("%s/Structure/query.json?" +\
                      "criteria=[graph_id$eq%d]") \
                      % (API_PATH, GRAPH_ID)

# Make a query to the API via a URL.
def QueryAPI(url):
    start_row = 0
    num_rows = 2000
    total_rows = -1
    rows = []
    done = False

    # The ontology has to be downloaded in pages, since the API will not return
    # more than 2000 rows at once.
    while not done:
        pagedUrl = url + '&start_row=%d&num_rows=%d' % (start_row,num_rows)

        print(pagedUrl)
        source = urllib.urlopen(pagedUrl).read()

        response = json.loads(source)
        rows += response['msg']
        
        if total_rows < 0:
            total_rows = int(response['total_rows'])

        start_row += len(response['msg'])

        if start_row >= total_rows:
            done = True

    print('Number of results: {}'.format(total_rows))
    return rows

# Download the first `n` data sets.  For negative `n` , download them all.
def DownloadDataSetList(n):
    dataSets = QueryAPI(DATA_SET_QUERY_URL)
  
    if n <= 0:
        return dataSets
    else:
        n = min(len(dataSets), n)

    return dataSets[:n]


# Download the mouse brain structures in a structure graph.
def DownloadStructures():
    structs = QueryAPI(STRUCTURES_URL)

    # Build a dict from structure id to structure and identify each node's 
    # direct descendants.
    structHash = {}
    for s in structs:
        s['num_children'] = 0
        s['structure_id_path'] = [int(sid) for sid in s['structure_id_path'].split('/') if sid != '']
        structHash[s['id']] = s 

    for sid,s in structHash.iteritems():
        if len(s['structure_id_path']) > 1:
            parentId = s['structure_id_path'][-2]
            structHash[parentId]['num_children'] += 1

    ## pull out the structure ids for structures in this structure graph that
    ## have no children (i.e. just the leaves)
    ## corrStructIds = [sid for sid,s in structHash.iteritems() if s['num_children'] == 0]
    # RB: no, leave all structures in and filter later
    corrStructIds = structHash.keys()

    return sorted(corrStructIds), structHash

def DownloadUnionizedData(dataSets):
    unionizes = [QueryAPI(UNIONIZE_FMT % (API_PATH,d['id'])) for d in dataSets]
    return unionizes

# Unionizes connectivity data for the adult mouse
# brain and transform it into useful data structures. 
def CreateConnectivityMatrix(dataSets,structureIds,structHash,unionizes):
    # Each injection experiment will have a connectivity vector.  This vector will be as long
    # as the number of requested structures.
    nstructs = len(structureIds)
    ndata = len(unionizes)
    print('ndata {} ndatasets {}'.format(ndata,len(dataSets)))

    sidHash = dict([(id,i) for (i,id) in enumerate(structureIds)])
    didHash = dict([(d['id'],i) for (i,d) in enumerate(dataSets)])
    
    connectivityL = numpy.empty([nstructs,ndata])
    connectivityL.fill(numpy.nan)
    connectivityR = numpy.empty([nstructs,ndata])
    connectivityR.fill(numpy.nan)

    # For each data set's set of unionizes, then for each individual structure,
    # fill in the structure's connectivity vector.
    for i,us in enumerate(unionizes):
        # for each unionize 
        for j,u in enumerate(us):
            sid = u['structure_id']
            did = u['section_data_set_id']

            struct = structHash[sid]
            struct['volume'] = u['sum_pixels']
            
            if i ==0 and j == 0:
              print(u)

            if sidHash.has_key(sid) and didHash.has_key(did):
                if u['hemisphere_id'] is 1:
                    connectivityL[sidHash[sid]][didHash[did]] = u['normalized_projection_volume']
                elif u['hemisphere_id'] is 2:
                    connectivityR[sidHash[sid]][didHash[did]] = u['normalized_projection_volume']
                elif u['hemisphere_id'] is 3:
                  pass
                  # this is just the average value of L+R
            else:
                print("ERROR: structure {}/injection {} skipped.".format(sid,did))
                
    return connectivityL, connectivityR


# Handle command line arguments. Usage is:
# `download_data.py <prefix>.json <nprobes>`

nargs = len(sys.argv)

fname = sys.argv[1] if nargs > 1 else "out.json"
n = int(sys.argv[2]) if nargs > 2 else 0

base,ext = os.path.splitext(fname)

structuresfile = base + "_structures" + ext
datasetsfile = base + "_datasets" + ext
unionizesfile = base + "_unionizes" + ext
connectivityLfile = base + "_connectivityL" + ext
connectivityRfile = base + "_connectivityR" + ext

connectivityL_csv = base + "_connectivityL" + '.csv'
connectivityR_csv = base + "_connectivityR" + '.csv'
structures_csv = base + "_structures" + '.csv'
injections_csv = base + "_injections" + '.csv'

# Download the list of data sets
dataSets = DownloadDataSetList(n)

with open(datasetsfile,"w") as fp:
  fp.write(json.dumps(dataSets))

with open(injections_csv,"w") as fp:
  M = []
  for v in dataSets:
    specimen = v['specimen']
    si = specimen['stereotaxic_injections']
    if len(si) != 1:
      print('Warning: invalid number of stereotaxic injections ({}). Discarding data.'.format(len(si)))
    else:
      prim = si[0]['primary_injection_structure']
      M.append([si[0]['id'],prim['acronym'],prim['name'],v['id']])
  w = csv.writer(fp,delimiter=',',quotechar='"',quoting=csv.QUOTE_MINIMAL)
  w.writerow(['injection_id','region_acronym','region_name','dataset_id'])
  for line in M: w.writerow(line)

# Download the list of structures
structureIds, structHash = DownloadStructures()

with open(structuresfile,"w") as fp:
  fp.write(json.dumps(structHash.values()))

with open(structures_csv,"w") as fp:
  M = []
  for sid in structureIds:
    v = structHash[sid]
    M.append([v['id'],v['acronym'],v['name'],v['parent_structure_id'],v['color_hex_triplet']])
  w = csv.writer(fp,delimiter=',',quotechar='"',quoting=csv.QUOTE_MINIMAL)
  w.writerow(['id','acronym','name','parent_structure_id','color_hex_triplet'])
  for line in M: w.writerow(line)

# Download the per-structure unionized data
try:
  with open(unionizesfile,"r") as fp:
    unionizes = json.load(fp)
except BaseException as E:
  unionizes = DownloadUnionizedData(dataSets)
  with open(unionizesfile,"w") as fp:
    fp.write(json.dumps(unionizes))

# Generate the connectivity matrix
connectivityL,connectivityR = CreateConnectivityMatrix(dataSets,structureIds,structHash,unionizes)

with open(connectivityLfile,"w") as fp:
  fp.write(json.dumps(connectivityL.tolist()))

with open(connectivityRfile,"w") as fp:
  fp.write(json.dumps(connectivityR.tolist()))

with open(connectivityL_csv,"w") as fp:
  M = connectivityL.tolist()
  fp.write('\n'.join([','.join([str(v) for v in line]) for line in M]))

with open(connectivityR_csv,"w") as fp:
  M = connectivityR.tolist()
  fp.write('\n'.join([','.join([str(v) for v in line]) for line in M]))
